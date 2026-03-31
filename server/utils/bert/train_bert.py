import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

DATA_PATH = "/kaggle/input/datasets/namish17/reviews/reviews.csv"
SAVE_DIR  = "/kaggle/working/bert_model"
# ────────────────────────────────────────────
EPOCHS      = 7
BATCH_SIZE  = 16
MAX_LEN     = 256
LR          = 2e-5
RANDOM_SEED = 42
PATIENCE    = 2     # early stopping patience


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts.reset_index(drop=True)
        self.ratings   = ratings.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rating":  torch.tensor(float(self.ratings[idx]), dtype=torch.float),
            "labels":  torch.tensor(int(self.labels[idx]),   dtype=torch.long),
        }


# ─────────────────────────────────────────────
# CONSISTENCY LOSS
# Penalises OR prediction with low stars, and CG prediction with high stars
# ─────────────────────────────────────────────
class ConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 0.3):
        super().__init__()
        self.weight = weight

    def forward(self, logits, ratings):
        probs    = torch.softmax(logits, dim=1)
        or_probs = probs[:, 1]   # probability of OR (real)
        cg_probs = probs[:, 0]   # probability of CG (fake)

        # OR + low star rating = suspicious
        low_rating_penalty  = or_probs * (1.0 - ratings)
        # CG + high star rating = suspicious
        high_rating_penalty = cg_probs * ratings

        return self.weight * (low_rating_penalty + high_rating_penalty).mean()


# ─────────────────────────────────────────────
# MODEL
# - rating_encoder: 2-layer MLP  Linear(1→16) → ReLU → Linear(16→32) → ReLU
# - classifier:     Linear(800→256) → ReLU → Dropout → Linear(256→2)
# ─────────────────────────────────────────────
class BertFakeReviewModel(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size   # 768

        # 2-layer rating encoder: 1 → 16 → 32
        self.rating_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden + 32, 256),        # 768 + 32 = 800
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, rating):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output                          # (B, 768)
        rating_emb = self.rating_encoder(rating.unsqueeze(1))      # (B, 32)
        combined   = torch.cat([cls_output, rating_emb], dim=1)    # (B, 800)
        return self.classifier(combined)                            # (B, 2)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH)

    # ── normalise columns ───────────────────
    df = df[["text_", "label", "rating"]].dropna().drop_duplicates()
    df["label"]  = df["label"].str.strip().str.upper()
    df["rating"] = df["rating"] / 5.0          # scale to [0, 1]

    label_mapping = {"CG": 0, "OR": 1}
    df["label"]   = df["label"].map(label_mapping)

    if df["label"].isnull().any():
        raise ValueError("Unknown labels found – expected only 'CG' and 'OR'.")

    df["label"] = df["label"].astype(int)
    print("Label distribution:\n", df["label"].value_counts())

    # ── class weights for imbalanced data ───
    n_cg = (df["label"] == 0).sum()
    n_or = (df["label"] == 1).sum()
    print(f"Class counts — CG: {n_cg}, OR: {n_or}")

    # ── train / val / test split ─────────────
    # split off 20% test
    train_val_texts, test_texts, train_val_labels, test_labels, train_val_ratings, test_ratings = (
        train_test_split(
            df["text_"], df["label"], df["rating"],
            test_size=0.2,
            stratify=df["label"],
            random_state=RANDOM_SEED,
        )
    )
    # split remaining into 87.5% train + 12.5% val → overall 70/10/20
    train_texts, val_texts, train_labels, val_labels, train_ratings, val_ratings = (
        train_test_split(
            train_val_texts, train_val_labels, train_val_ratings,
            test_size=0.125,
            stratify=train_val_labels,
            random_state=RANDOM_SEED,
        )
    )

    print(f"Split — train: {len(train_texts)}, val: {len(val_texts)}, test: {len(test_texts)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_ds = ReviewDataset(train_texts, train_ratings, train_labels, tokenizer)
    val_ds   = ReviewDataset(val_texts,   val_ratings,   val_labels,   tokenizer)
    test_ds  = ReviewDataset(test_texts,  test_ratings,  test_labels,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── model ───────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertFakeReviewModel().to(device)

    # ── weighted cross-entropy loss ──────────
    class_weights = torch.tensor([1.0/n_cg, 1.0/n_or], dtype=torch.float)
    class_weights = (class_weights / class_weights.sum()).to(device)
    loss_fn     = nn.CrossEntropyLoss(weight=class_weights)
    consistency = ConsistencyLoss(weight=0.3)

    # ── layer-wise learning rates ────────────
    optimizer_grouped_parameters = [
        # lower BERT layers — smallest LR (basic syntax, needs least updating)
        {
            "params": model.bert.encoder.layer[:6].parameters(),
            "lr": LR * 0.1,
        },
        # upper BERT layers — medium LR (semantics)
        {
            "params": model.bert.encoder.layer[6:].parameters(),
            "lr": LR * 0.5,
        },
        # BERT embeddings + pooler
        {
            "params": list(model.bert.embeddings.parameters())
                    + list(model.bert.pooler.parameters()),
            "lr": LR * 0.1,
        },
        # new heads — full LR
        {"params": model.classifier.parameters(),     "lr": LR},
        {"params": model.rating_encoder.parameters(), "lr": LR},
    ]
    optim = AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── freeze BERT for epoch 1 ──────────────
    for param in model.bert.parameters():
        param.requires_grad = False
    print("BERT frozen for epoch 1")

    # ── training loop ───────────────────────
    best_val_loss = float("inf")
    no_improve    = 0
    history       = []
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):

        # unfreeze BERT from epoch 2
        if epoch == 2:
            for param in model.bert.parameters():
                param.requires_grad = True
            print("BERT unfrozen")

        # ── train ──
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optim.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rating         = batch["rating"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, rating)
            loss   = loss_fn(logits, labels) + consistency(logits, rating)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch}/{EPOCHS}  step {step}/{len(train_loader)}  "
                      f"train_loss={total_train_loss/step:.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ── validate ──
        model.eval()
        total_val_loss = 0.0
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rating         = batch["rating"].to(device)
                labels         = batch["labels"].to(device)

                logits = model(input_ids, attention_mask, rating)
                loss   = loss_fn(logits, labels) + consistency(logits, rating)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc      = accuracy_score(val_true, val_preds)

        print(f"Epoch {epoch}/{EPOCHS} — "
              f"train_loss: {avg_train_loss:.4f}  "
              f"val_loss: {avg_val_loss:.4f}  "
              f"val_acc: {val_acc:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "val_loss":   avg_val_loss,
            "val_acc":    val_acc,
        })

        # ── checkpoint + early stopping ──────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve    = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "bert_best.pt"))
            print(f"   ↳ New best checkpoint saved (val_loss={avg_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"   ↳ No improvement ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print("⏹  Early stopping triggered.")
                break

    # ── evaluation on test set ───────────────
    print("\nEvaluating on test set …")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "bert_best.pt"),
                                     map_location=device))
    model.eval()

    all_preds, all_labels_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rating         = batch["rating"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, rating)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    acc    = accuracy_score(all_labels_list, all_preds)
    report = classification_report(
        all_labels_list, all_preds, target_names=["CG (Fake)", "OR (Real)"]
    )

    print(f"Test Accuracy : {acc:.4f}")
    print(report)

    final_pt = os.path.join(SAVE_DIR, "bert.pt")
    if os.path.exists(final_pt):
        os.remove(final_pt)
    os.rename(os.path.join(SAVE_DIR, "bert_best.pt"), final_pt)

    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "tokenizer"))

    meta = {
        "max_len":       MAX_LEN,
        "label_mapping": {"CG": 0, "OR": 1},
        "id2label":      {0: "CG", 1: "OR"},
        "accuracy":      acc,
        "history":       history,
    }
    with open(os.path.join(SAVE_DIR, "model_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\n All artifacts saved to {SAVE_DIR}")


if __name__ == "__main__":
    train()
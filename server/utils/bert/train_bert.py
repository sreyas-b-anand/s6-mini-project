import os
import pickle
import numpy as np
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

# ── UPDATE THESE PATHS BEFORE RUNNING ───────
DATA_PATH = "/kaggle/input/datasets/sreyasbanand/reviews-dataset-new/reviews.csv"   # <-- update this
SAVE_DIR  = "/kaggle/working/bert_model"                    # <-- update if needed
# ────────────────────────────────────────────
EPOCHS      = 3
BATCH_SIZE  = 16
MAX_LEN     = 256
LR          = 2e-5
RANDOM_SEED = 42


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
            # rating already normalised to [0,1] before dataset creation
            "rating":  torch.tensor(float(self.ratings[idx]), dtype=torch.float),
            "labels":  torch.tensor(int(self.labels[idx]),   dtype=torch.long),
        }


# ─────────────────────────────────────────────
# MODEL  — rating gets its own projection branch
#          so it isn't drowned out by 768 BERT dims
# ─────────────────────────────────────────────
class BertFakeReviewModel(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size   # 768

        # project the single rating scalar → 32-dim embedding
        self.rating_proj = nn.Sequential(
            nn.Linear(1, 32),
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
        cls_output = outputs.pooler_output                              # (B, 768)

        # rating: (B,) → (B, 1) → (B, 32)
        rating_emb = self.rating_proj(rating.unsqueeze(1))             # (B, 32)

        combined = torch.cat([cls_output, rating_emb], dim=1)          # (B, 800)
        return self.classifier(combined)                                # (B, 2)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    print("🚀  Loading data …")
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

    # ── split ───────────────────────────────
    train_texts, test_texts, train_labels, test_labels, train_ratings, test_ratings = (
        train_test_split(
            df["text_"], df["label"], df["rating"],
            test_size=0.2,
            stratify=df["label"],
            random_state=RANDOM_SEED,
        )
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_ds = ReviewDataset(train_texts, train_ratings, train_labels, tokenizer)
    test_ds  = ReviewDataset(test_texts,  test_ratings,  test_labels,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── model / optimiser ───────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model   = BertFakeReviewModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── training loop ───────────────────────
    best_val_loss = float("inf")
    history       = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optim.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rating         = batch["rating"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, rating)
            loss   = loss_fn(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"  Epoch {epoch}/{EPOCHS}  step {step}/{len(train_loader)}  "
                      f"loss={total_loss/step:.4f}")

        avg_loss = total_loss / len(train_loader)
        history.append({"epoch": epoch, "train_loss": avg_loss})
        print(f"✅  Epoch {epoch}/{EPOCHS} complete – avg loss: {avg_loss:.4f}")

        # save best checkpoint
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "bert_best.pt"))
            print(f"   ↳ New best checkpoint saved (loss={avg_loss:.4f})")

    # ── evaluation ──────────────────────────
    print("\n📊  Evaluating …")
    best_ckpt = os.path.join(SAVE_DIR, "bert_best.pt")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
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

    print(f"Accuracy : {acc:.4f}")
    print(report)

    # ── save final model ────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    # safe rename: remove destination first to avoid os.rename failing on Windows
    final_pt = os.path.join(SAVE_DIR, "bert.pt")
    if os.path.exists(final_pt):
        os.remove(final_pt)
    os.rename(best_ckpt, final_pt)

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

    print(f"\n✅  All artifacts saved to {SAVE_DIR}")


if __name__ == "__main__":
    train()
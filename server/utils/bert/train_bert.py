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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "reviews.csv")
SAVE_DIR  = os.path.join(BASE_DIR, "model", "bert_model_new")
EPOCHS      = 3
BATCH_SIZE  = 16
MAX_LEN     = 256
LR          = 2e-5
RANDOM_SEED = 42

class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, labels, tokenizer, max_len=MAX_LEN):
        self.texts    = texts.reset_index(drop=True)
        self.ratings  = ratings.reset_index(drop=True)
        self.labels   = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len  = max_len

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
            "input_ids":      encoding["input_ids"].squeeze(0),          # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),     # (seq_len,)
            "rating":         torch.tensor(float(self.ratings[idx]), dtype=torch.float),
            "labels":         torch.tensor(int(self.labels[idx]),   dtype=torch.long),
        }

# ─────────────────────────────────────────────
# MODEL  (BERT + rating scalar → 2-class head)
# ─────────────────────────────────────────────
class BertFakeReviewModel(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden    = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, rating):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output                          # (B, 768)
        combined   = torch.cat([cls_output, rating.unsqueeze(1)], dim=1)  # (B, 769)
        return self.classifier(combined)                            # (B, 2)

# ─────────────────────────────────────────────
# INFERENCE HELPER  (used by FastAPI)
# ─────────────────────────────────────────────
def predict(texts: list[str], ratings: list[float], model_dir: str = SAVE_DIR) -> list[dict]:
    """
    texts   : list of review strings
    ratings : list of star ratings (1-5)
    returns : list of {"label": "OR"|"CG", "confidence": float}
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model     = BertFakeReviewModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, "bert.pt"), map_location=device))
    model.to(device).eval()

    id2label  = {0: "CG", 1: "OR"}
    results   = []

    with torch.no_grad():
        for text, rating in zip(texts, ratings):
            enc = tokenizer(
                str(text),
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            r              = torch.tensor([[rating / 5.0]], dtype=torch.float).squeeze(1).to(device)

            logits = model(input_ids, attention_mask, r)
            probs  = torch.softmax(logits, dim=1)
            pred   = torch.argmax(probs, dim=1).item()
            conf   = probs[0][pred].item()

            results.append({"label": id2label[pred], "confidence": round(conf, 4)})

    return results

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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── model / optimiser ───────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model   = BertFakeReviewModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.1 * total_steps),   # 10 % warmup
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clip
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
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "bert_best.pt")))  # use best weights
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

    acc = accuracy_score(all_labels_list, all_preds)
    report = classification_report(all_labels_list, all_preds, target_names=["CG (Fake)", "OR (Real)"])

    print(f"Accuracy : {acc:.4f}")
    print(report)

    # ── save everything ─────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    # rename best → final
    os.rename(
        os.path.join(SAVE_DIR, "bert_best.pt"),
        os.path.join(SAVE_DIR, "bert.pt"),
    )

    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "tokenizer"))

    # metadata / config pkl (useful for the FastAPI server)
    meta = {
        "max_len":       MAX_LEN,
        "label_mapping": {"CG": 0, "OR": 1},
        "id2label":      {0: "CG", 1: "OR"},
        "accuracy":      acc,
        "history":       history,
    }
    with open(os.path.join(SAVE_DIR, "model_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    


if __name__ == "__main__":
    train()
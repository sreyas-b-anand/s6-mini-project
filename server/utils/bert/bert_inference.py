import os
import pickle
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


# ─────────────────────────────────────────────
# MODEL  (must match bert_train.py exactly)
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
# INFERENCE CLASS  (load once, reuse forever)
# ─────────────────────────────────────────────
class FakeReviewDetector:
    def __init__(self, model_dir: str):
        """
        Load model + tokenizer + meta once at startup.
        model_dir must contain:
            bert.pt, tokenizer/, model_meta.pkl
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── load meta ──
        meta_path = os.path.join(model_dir, "model_meta.pkl")
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        self.max_len  = self.meta["max_len"]
        self.id2label = self.meta["id2label"]   # {0: "CG", 1: "OR"}

        # ── load tokenizer ──
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer")
        )

        # ── load model weights ──
        self.model = BertFakeReviewModel()
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "bert.pt"),
                map_location=self.device,
            )
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"✅  FakeReviewDetector ready on {self.device}")

    # ─────────────────────────────────────────
    def _encode(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    # ─────────────────────────────────────────
    def predict(self, texts: list[str], ratings: list[float]) -> list[dict]:
        """
        texts   : list of review strings
        ratings : list of raw star ratings (1–5)
        returns : list of {"label": "OR"|"CG", "confidence": float}
        """
        if len(texts) != len(ratings):
            raise ValueError("texts and ratings must have the same length.")

        all_input_ids       = []
        all_attention_masks = []
        all_ratings         = []

        for text, rating in zip(texts, ratings):
            enc = self._encode(str(text))
            all_input_ids.append(enc["input_ids"])
            all_attention_masks.append(enc["attention_mask"])
            all_ratings.append(rating / 5.0)    # normalise to [0, 1]

        input_ids      = torch.stack(all_input_ids).to(self.device)
        attention_mask = torch.stack(all_attention_masks).to(self.device)
        ratings_tensor = torch.tensor(all_ratings, dtype=torch.float).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, ratings_tensor)  # (B, 2)
            probs  = torch.softmax(logits, dim=1)                           # (B, 2)
            preds  = torch.argmax(probs, dim=1)                             # (B,)

        results = []
        for pred, prob in zip(preds.cpu().tolist(), probs.cpu().tolist()):
            results.append({
                "label":      self.id2label[pred],
                "confidence": round(prob[pred], 4),
            })

        return results
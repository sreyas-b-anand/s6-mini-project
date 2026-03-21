# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
# from torch.optim import AdamW
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA_PATH = os.path.join(BASE_DIR, "..", "data", "reviews.csv")
# SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "bert_new_one")

# class ReviewDataset(Dataset):
#     def __init__(self, texts, ratings, labels, tokenizer, max_len=256):
#         self.texts = texts.tolist()
#         self.ratings = ratings.tolist()
#         self.labels = labels.tolist()
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             str(self.texts[idx]),
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'rating': torch.tensor(self.ratings[idx], dtype=torch.float),
#             'labels': torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# class Bert(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")

#         self.fc = nn.Sequential(
#             nn.Linear(768 + 1, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 2)
#         )

#     def forward(self, input_ids, attention_mask, rating):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = outputs.pooler_output

#         combined = torch.cat([cls_output, rating.unsqueeze(1)], dim=1)
#         return self.fc(combined)
# def train():

#     print("🚀 Starting BERT Training...")

#     # -------------------------
#     # LOAD DATA
#     # -------------------------
#     df = pd.read_csv(DATA_PATH)
#     df = df[['text_', 'label', 'rating']]

#     df['rating'] = df['rating'] / 5.0
#     df = df.dropna().drop_duplicates()

#     df['label'] = df['label'].str.strip().str.upper()

#     label_mapping = {
#         'CG': 0,
#         'OR': 1
#     }

#     df['label'] = df['label'].map(label_mapping)

#     if df['label'].isnull().any():
#         raise ValueError("Unexpected label found!")

#     df['label'] = df['label'].astype(int)

#     print("\nLabel distribution:")
#     print(df['label'].value_counts())

#     # -------------------------
#     # SPLIT
#     # -------------------------
#     train_texts, test_texts, train_labels, test_labels, train_ratings, test_ratings = train_test_split(
#         df['text_'],
#         df['label'],
#         df['rating'],
#         test_size=0.2,
#         stratify=df['label'],
#         random_state=42
#     )


#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    
#     train_dataset = ReviewDataset(train_texts, train_ratings, train_labels, tokenizer)
#     test_dataset = ReviewDataset(test_texts, test_ratings, test_labels, tokenizer)

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=16)

    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = Bert().to(device)

#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = AdamW(model.parameters(), lr=2e-5)

#     epochs = 3
#     total_steps = len(train_loader) * epochs

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=0,
#         num_training_steps=total_steps
#     )

    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         print(f"\nEpoch {epoch+1}/{epochs}")

#         for batch in train_loader:
#             optimizer.zero_grad()

#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             rating = batch['rating'].to(device)
#             labels = batch['labels'].to(device)

#             logits = model(input_ids, attention_mask, rating)

#             loss = loss_fn(logits, labels)
#             total_loss += loss.item()

#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         print(f"Loss: {total_loss / len(train_loader):.4f}")

    
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             rating = batch['rating'].to(device)
#             labels = batch['labels'].to(device)

#             logits = model(input_ids, attention_mask, rating)
#             preds = torch.argmax(logits, dim=1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     print("\nEvaluation Results")
#     print("Accuracy:", accuracy_score(all_labels, all_preds))
#     print(classification_report(all_labels, all_preds))

#     # -------------------------
#     # SAVE MODEL
#     # -------------------------
#     os.makedirs(SAVE_PATH, exist_ok=True)

#     torch.save(model.state_dict(), os.path.join(SAVE_PATH, "bert.pt"))
#     tokenizer.save_pretrained(os.path.join(SAVE_PATH, "tokenizer"))

#     print("\n✅ Model Saved Successfully")


# if __name__ == "__main__":
#     train()


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import os

# =========================
# 1️⃣ Load Dataset
# =========================


class ReviewDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =========================
# 5️⃣ DataLoaders
# =========================


def train():
    df = pd.read_csv("../data/reviews.csv")

    df = df[['text_', 'label']]
    df = df.dropna().drop_duplicates()

    # 🔥 Convert CG / OR → 0 / 1
    df['label'] = df['label'].str.strip().str.upper()

    label_mapping = {
        'CG': 0,   # Fake
        'OR': 1    # Original / Real
    }

    df['label'] = df['label'].map(label_mapping)

    # Safety check
    if df['label'].isnull().any():
        raise ValueError("Unexpected label found in dataset!")

    df['label'] = df['label'].astype(int)

    print("Label distribution:")
    print(df['label'].value_counts())

    # =========================
    # 2️⃣ Train Test Split
    # =========================

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text_'],
        df['label'],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # =========================
    # 6️⃣ Model
    # =========================

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.backends.cudnn.benchmark = True 
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # =========================
    # 7️⃣ Training Loop
    # =========================

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print("epoch" , epoch)
        for batch in train_loader:
            print("epoch inside nested" , epoch)

            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        model.save_pretrained(f"../models/checkpoint_epoch_{epoch+1}")

    # =========================
    # 8️⃣ Evaluation
    # =========================

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nEvaluation Results")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

    # =========================
    # 9️⃣ Save Model
    # =========================

    save_path = "../models/bert_fake_review_model"
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("\nModel Saved Successfully")


if __name__ == "__main__":
    train()
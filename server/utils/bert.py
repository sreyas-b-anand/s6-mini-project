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

# =========================
# 3️⃣ Tokenizer
# =========================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =========================
# 4️⃣ Dataset Class
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
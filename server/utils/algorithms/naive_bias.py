import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

import joblib

# ─────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────
csv_path = "/kaggle/input/datasets/nevinjosephantony/reviews/reviews.csv"
dataset = pd.read_csv(csv_path)
dataset.columns = dataset.columns.str.strip()

required_cols = ['category', 'rating', 'text_', 'label']
for col in required_cols:
    if col not in dataset.columns:
        raise ValueError(f"Missing column: {col}")

dataset = dataset.dropna(subset=required_cols)

dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
dataset = dataset.dropna(subset=['rating'])

# ─────────────────────────────────────────────────────────────
def train_model():
    X = dataset[['category', 'rating', 'text_']]
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # Preprocessing
    clt = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(), 'text_'),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), ['category']),
            ('num', MinMaxScaler(), ['rating']),
        ],
        remainder='drop'
    )

    model_pipeline = Pipeline([
        ('preprocess', clt),
        ('nb', MultinomialNB()),
    ])

    # Full grid (no reduction)
    param_grid = {
        'nb__alpha': [0.1, 0.5, 1.0, 2.0],
        'nb__fit_prior': [True, False],
        'preprocess__tfidf__max_features': [10000, 30000],
        'preprocess__tfidf__ngram_range': [(1, 1), (1, 2)],
        'preprocess__tfidf__sublinear_tf': [True, False],
    }

    grid = list(ParameterGrid(param_grid))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    total_runs = len(grid) * 5
    print(f"\nTotal fits: {total_runs}\n")

    best_score = -1
    best_params = None
    best_pipeline = None

    # Progress bar
    with tqdm(total=total_runs, desc="Training Progress") as pbar:
        for params in grid:
            model_pipeline.set_params(**params)

            scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_tr = X_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_tr = y_train_enc[train_idx]
                y_val = y_train_enc[val_idx]

                model_pipeline.fit(X_tr, y_tr)
                preds = model_pipeline.predict(X_val)

                score = f1_score(y_val, preds, average='weighted')
                scores.append(score)

                pbar.update(1)

            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_pipeline = Pipeline(model_pipeline.steps)

    print("\n── Best Results ──────────────────────────")
    print(f"Best F1 score   : {best_score:.4f}")
    print("Best parameters :")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Final training on full train set
    best_pipeline.fit(X_train, y_train_enc)

    # Evaluation
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred  = best_pipeline.predict(X_test)

    print("\n── Evaluation ───────────────────────────")
    print("Train Accuracy :", accuracy_score(y_train_enc, y_train_pred))
    print("Test  Accuracy :", accuracy_score(y_test_enc, y_test_pred))

    print("\nClassification Report:\n")
    print(classification_report(
        y_test_enc,
        y_test_pred,
        target_names=le.classes_
    ))

    return best_pipeline, le


# ─────────────────────────────────────────────────────────────
def save_model(model_pipeline, le):
    MODEL_PATH = "/kaggle/working/naive_bayes_pipeline.pkl"

    joblib.dump({
        "model": model_pipeline,
        "label_encoder": le,
    }, MODEL_PATH)

    print(f"\nModel saved successfully → {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, le = train_model()
    save_model(model, le)
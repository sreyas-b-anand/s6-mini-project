import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

print("Starting XGBoost training script")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Base directory:", BASE_DIR)

csv_path = os.path.join(BASE_DIR, '../..', 'data', 'reviews.csv')
print("Loading dataset from:", csv_path)

dataset = pd.read_csv(csv_path)
print("Dataset loaded successfully")
print("Dataset shape:", dataset.shape)


def train_model():
    print("Preparing features and labels")

    X = dataset[['category', 'rating', 'text_']]
    y = dataset['label']

    print("Features and labels separated")

    print("Performing train-test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train-test split completed")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("Encoding labels")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print("Label encoding completed")
    print("Classes:", le.classes_)

    print("Building preprocessing pipeline")

    clt = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2)
            ), "text_"),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["category"]),
            ('num', 'passthrough', ["rating"]),
        ],
        remainder='drop'
    )

    print("Preprocessing pipeline created")

    print("Initializing XGBoost model")

    xgb_model = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=1
    )

    print("XGBoost initialized")

    param_grid = {
        'xgb__n_estimators': [100, 300],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__max_depth': [4, 6],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }

    print("Constructing full pipeline")

    model_pipeline = Pipeline([
        ('preprocess', clt),
        ('xgb', xgb_model)
    ])

    print("Pipeline constructed")

    print("Initializing GridSearchCV")

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    print("Training model with GridSearchCV")

    grid.fit(X_train, y_train)

    print("Grid search completed")

    print("Best Parameters:", grid.best_params_)
    print("Best Cross Validation Score:", grid.best_score_)

    best_model = grid.best_estimator_

    print("Evaluating model")

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print("Results")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    print("Classification Report")
    print(classification_report(y_test, y_test_pred))

    return best_model, le


def save_model(model_pipeline, le):

    print("Saving model")

    models_dir = os.path.join(BASE_DIR, '../..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'xgb_pipeline.pkl')

    joblib.dump({
        "model": model_pipeline,
        "label_encoder": le
    }, model_path)

    print("Model saved successfully at:", model_path)


if __name__ == "__main__":

    model, le = train_model()
    #save_model(model, le)

    print("Script finished successfully")
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = ''

csv_path = os.path.join(BASE_DIR, '../..', 'data', 'reviews.csv')
dataset = pd.read_csv(csv_path)


def train_model():
    X = dataset[['category', 'rating', 'text_']]
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test) # CG -> 0 , OR -> 1

    clt = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1,2)), "text_"),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["category"]),
            ('num', 'passthrough', ["rating"]),
        ],
        remainder='drop'
    )

    model_pipeline = Pipeline([
        ('preprocess', clt),
        ('svm', LinearSVC(random_state=42, C=1))
    ])

    model_pipeline.fit(X_train, y_train)
    # y_pred = model_pipeline.predict(X_test)

    # y_train_pred = model_pipeline.predict(X_train)
    # y_test_pred  = model_pipeline.predict(X_test)

    # print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    # print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    

    # print(classification_report(y_test, y_test_pred))
    
    return model_pipeline , le
    
# use this in the other files to save model as .pkl file
    
def save_model(model_pipeline , le):
    models_dir = os.path.join(BASE_DIR, '../..', 'models')
    os.makedirs(models_dir, exist_ok=True)

   
    MODEL_PATH = os.path.join(models_dir, 'svm_pipeline.pkl')
    joblib.dump({
        "model": model_pipeline,
        "label_encoder": le
    }, MODEL_PATH)
    
    print("Model saved successfully!")
    
    

if __name__=="__main__":
    model , le = train_model()
    save_model(model  , le)
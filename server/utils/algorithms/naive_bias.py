import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
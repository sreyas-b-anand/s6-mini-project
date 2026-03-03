
# ==================================================
# IMPORTS
# ==================================================
import joblib
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class GraphFeatureExtractor:
    def __init__(self):
        self.graph = None
        self.vectorizer = None
        self.review_data = None
        self.suspicion_scores = None

    
    def build_reference_graph(self, dataset_path):
        print("Building reference graph...")

        df = pd.read_csv(dataset_path)
        print("Original dataset size:", len(df))

        # Balanced sampling (500 fake + 500 real)
        df_fake = df[df['label'] == "OR"].sample(500, random_state=42)
        df_real = df[df['label'] == "CG"].sample(500, random_state=42)

        df = pd.concat([df_fake, df_real]).reset_index(drop=True)
        print("Sampled dataset size:", len(df))

        df = self._prepare_data(df)

        print("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(max_features=800, stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(df['cleaned_text'])

        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(tfidf_matrix)

        print("Building graph...")
        self.graph = self._build_review_graph(df, similarity_matrix)

        print("Calculating suspicion scores...")
        self.suspicion_scores = self._calculate_suspicion_scores(self.graph, df)

        df['graph_suspicion_score'] = df.index.map(
            lambda x: self.suspicion_scores.get(x, 0)
        )

        self.review_data = df
        print("Graph built successfully ✅")

    
    def _prepare_data(self, df):
        df = df.copy()

        df['cleaned_text'] = df['text_'].apply(self._clean_text)
        df['sentiment'] = df['cleaned_text'].apply(self._get_sentiment)

        df['rating_sentiment_diff'] = abs(
            df['rating'] - (df['sentiment'] * 4 + 1)
        )

        return df

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)

    def _get_sentiment(self, text):
        try:
            blob = TextBlob(text)
            return (blob.sentiment.polarity + 1) / 2
        except:
            return 0.5

    
    def _build_review_graph(self, df, similarity_matrix):
        G = nx.Graph()

        for idx, row in df.iterrows():
            G.add_node(
                idx,
                text=row['cleaned_text'],
                rating=row['rating'],
                category=row['category'],
                sentiment=row['sentiment'],
                rating_sentiment_diff=row['rating_sentiment_diff']
            )

        n = len(df)

        for i in range(n):
            for j in range(i + 1, n):

                if similarity_matrix[i][j] > 0.7:
                    G.add_edge(i, j, weight=similarity_matrix[i][j], type='text_similarity')

                if (df.iloc[i]['rating_sentiment_diff'] > 3 and
                        df.iloc[j]['rating_sentiment_diff'] > 3):
                    G.add_edge(i, j, weight=0.8, type='inconsistency')

                if df.iloc[i]['category'] == df.iloc[j]['category']:
                    G.add_edge(i, j, weight=0.5, type='same_category')

        return G

    
    def _calculate_suspicion_scores(self, G, df):
        suspicion_scores = {}

        for node in G.nodes():
            score = 0
            edges = list(G.edges(node, data=True))

            score += len([e for e in edges if e[2]['type'] == 'text_similarity']) * 0.25
            score += len([e for e in edges if e[2]['type'] == 'inconsistency']) * 0.35
            score += len([e for e in edges if e[2]['type'] == 'same_category']) * 0.15

            if df.iloc[node]['rating_sentiment_diff'] > 3:
                score += 0.25

            suspicion_scores[node] = min(score, 1.0)

        return suspicion_scores

    def predict_review(self, review_text, rating, category):

        cleaned = self._clean_text(review_text)
        sentiment = self._get_sentiment(cleaned)
        rating_sentiment_diff = abs(rating - (sentiment * 4 + 1))

        new_vec = self.vectorizer.transform([cleaned])
        stored_vecs = self.vectorizer.transform(self.review_data['cleaned_text'])

        similarities = cosine_similarity(new_vec, stored_vecs)[0]

        suspicion_score = 0

        high_sim_count = np.sum(similarities > 0.7)
        suspicion_score += high_sim_count * 0.25

        if rating_sentiment_diff > 3:
            suspicion_score += 0.25

        same_category_count = np.sum(
            self.review_data['category'] == category
        )
        suspicion_score += same_category_count * 0.0005

        suspicion_score = min(suspicion_score, 1.0)

        label = "Fake" if suspicion_score > 0.5 else "Real"

        return {
            "prediction": label,
            "suspicion_score": float(suspicion_score),
            "similar_reviews_found": int(high_sim_count)
        }

    
    def save_model(self, path):
        joblib.dump(self, path)
        




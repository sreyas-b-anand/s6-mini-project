# ==================================================
# IMPORTS
# ==================================================
import joblib
import pandas as pd
import numpy as np
import networkx as nx
import re
import nltk
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import warnings
import os

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# ==================================================
# GPU CHECK
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==================================================
# DIRECTORY SETUP
# ==================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")


csv_path = os.path.join(BASE_DIR, '..', 'data', 'reviews.csv')
dataset = pd.read_csv(csv_path)



# ==================================================
# GRAPH FEATURE EXTRACTOR
# ==================================================

class GraphFeatureExtractor:

    def __init__(self):

        self.graph = None
        self.vectorizer = None
        self.review_data = None
        self.suspicion_scores = None


    # ==================================================
    # BUILD GRAPH FROM DATASET
    # ==================================================

    def build_reference_graph(self, dataset_path):

        print("Building reference graph...")

        df = pd.read_csv(dataset_path)

        print("Original dataset size:", len(df))

        # Balanced sampling
        df_fake = df[df['label'] == "OR"].sample(2500, random_state=42)
        df_real = df[df['label'] == "CG"].sample(2500, random_state=42)

        df = pd.concat([df_fake, df_real]).reset_index(drop=True)

        print("Sampled dataset size:", len(df))

        df = self._prepare_data(df)

        # ==================================================
        # TF-IDF
        # ==================================================

        print("Creating TF-IDF matrix...")

        self.vectorizer = TfidfVectorizer(
            max_features=800,
            stop_words='english'
        )

        tfidf_matrix = self.vectorizer.fit_transform(df['cleaned_text'])

        # Convert to GPU tensor
        tfidf_tensor = torch.tensor(
            tfidf_matrix.toarray(),
            dtype=torch.float32
        ).to(device)

        # ==================================================
        # COSINE SIMILARITY (GPU)
        # ==================================================

        print("Computing similarity matrix on GPU...")

        tfidf_norm = torch.nn.functional.normalize(tfidf_tensor, p=2, dim=1)

        similarity_matrix = torch.mm(
            tfidf_norm,
            tfidf_norm.T
        )

        similarity_matrix = similarity_matrix.cpu().numpy()

        # ==================================================
        # BUILD GRAPH
        # ==================================================

        print("Building graph...")

        self.graph = self._build_review_graph(df, similarity_matrix)

        print("Calculating suspicion scores...")

        self.suspicion_scores = self._calculate_suspicion_scores(
            self.graph,
            df
        )

        df['graph_suspicion_score'] = df.index.map(
            lambda x: self.suspicion_scores.get(x, 0)
        )

        self.review_data = df

        print("Graph built successfully")


    # ==================================================
    # PREPARE DATA
    # ==================================================

    def _prepare_data(self, df):

        df = df.copy()

        df['cleaned_text'] = df['text_'].apply(self._clean_text)

        df['sentiment'] = df['cleaned_text'].apply(
            self._get_sentiment
        )

        df['rating_sentiment_diff'] = abs(
            df['rating'] - (df['sentiment'] * 4 + 1)
        )

        return df


    # ==================================================
    # TEXT CLEANING
    # ==================================================

    def _clean_text(self, text):

        text = str(text).lower()

        text = re.sub(r'[^a-zA-Z\s]', '', text)

        words = text.split()

        stop_words = set(stopwords.words('english'))

        words = [
            w for w in words
            if w not in stop_words
        ]

        return ' '.join(words)


    # ==================================================
    # SENTIMENT
    # ==================================================

    def _get_sentiment(self, text):

        try:

            blob = TextBlob(text)

            return (blob.sentiment.polarity + 1) / 2

        except:

            return 0.5


    # ==================================================
    # FAST GRAPH BUILDING
    # ==================================================

    def _build_review_graph(self, df, similarity_matrix):

        G = nx.Graph()

        # Add nodes
        for idx, row in df.iterrows():

            G.add_node(
                idx,
                text=row['cleaned_text'],
                rating=row['rating'],
                category=row['category'],
                sentiment=row['sentiment'],
                rating_sentiment_diff=row['rating_sentiment_diff']
            )

        # ==================================================
        # TEXT SIMILARITY EDGES
        # ==================================================

        sim_indices = np.argwhere(similarity_matrix > 0.7)

        for i, j in sim_indices:

            if i < j:

                G.add_edge(
                    int(i),
                    int(j),
                    weight=float(similarity_matrix[i][j]),
                    type='text_similarity'
                )

        # ==================================================
        # INCONSISTENCY EDGES
        # ==================================================

        inconsistency_nodes = np.where(
            df['rating_sentiment_diff'] > 3
        )[0]

        for i in range(len(inconsistency_nodes)):
            for j in range(i + 1, len(inconsistency_nodes)):

                G.add_edge(
                    int(inconsistency_nodes[i]),
                    int(inconsistency_nodes[j]),
                    weight=0.8,
                    type='inconsistency'
                )

        # ==================================================
        # SAME CATEGORY EDGES
        # ==================================================

        categories = df['category'].values

        for cat in np.unique(categories):

            indices = np.where(categories == cat)[0]

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):

                    G.add_edge(
                        int(indices[i]),
                        int(indices[j]),
                        weight=0.5,
                        type='same_category'
                    )

        return G


    # ==================================================
    # SUSPICION SCORE
    # ==================================================

    def _calculate_suspicion_scores(self, G, df):

        suspicion_scores = {}

        for node in G.nodes():

            score = 0

            edges = list(G.edges(node, data=True))

            score += len([
                e for e in edges
                if e[2]['type'] == 'text_similarity'
            ]) * 0.25

            score += len([
                e for e in edges
                if e[2]['type'] == 'inconsistency'
            ]) * 0.35

            score += len([
                e for e in edges
                if e[2]['type'] == 'same_category'
            ]) * 0.15

            if df.iloc[node]['rating_sentiment_diff'] > 3:

                score += 0.25

            suspicion_scores[node] = min(score, 1.0)

        return suspicion_scores


    # ==================================================
    # PREDICT REVIEW
    # ==================================================

    def predict_review(self, review_text, rating, category):

        cleaned = self._clean_text(review_text)

        sentiment = self._get_sentiment(cleaned)

        rating_sentiment_diff = abs(
            rating - (sentiment * 4 + 1)
        )

        new_vec = self.vectorizer.transform([cleaned]).toarray()

        stored_vecs = self.vectorizer.transform(
            self.review_data['cleaned_text']
        ).toarray()

        new_tensor = torch.tensor(new_vec).to(device)
        stored_tensor = torch.tensor(stored_vecs).to(device)

        new_norm = torch.nn.functional.normalize(new_tensor, dim=1)
        stored_norm = torch.nn.functional.normalize(stored_tensor, dim=1)

        similarities = torch.mm(
            new_norm,
            stored_norm.T
        ).cpu().numpy()[0]

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


    # ==================================================
    # SAVE MODEL
    # ==================================================

    def save_model(self, filename="graph_model.pkl"):

        save_path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(self, save_path)

        print("Model saved to:", save_path)




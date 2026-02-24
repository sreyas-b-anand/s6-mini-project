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
import joblib
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class GraphFeatureExtractor:
    def __init__(self):
        self.graph = None
        self.vectorizer = None
        self.review_data = None
        self.suspicion_scores = None
        self.text_similarity_matrix = None
    
    def build_reference_graph(self):
        df = pd.read_csv("server/data/reviews.csv")
        # df = pd.read_csv(dataset_path)
        df = self._prepare_data(df)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(df['cleaned_text'])
        self.text_similarity_matrix = cosine_similarity(tfidf_matrix)
        self.graph = self._build_review_graph(df, tfidf_matrix)
        self.suspicion_scores = self._calculate_suspicion_scores(self.graph, df)
        df['graph_suspicion_score'] = df.index.map(lambda x: self.suspicion_scores.get(x, 0))
        self.review_data = df
        return self.graph
    
    def _prepare_data(self, df):
        df = df.copy()
        df['cleaned_text'] = df['text_'].apply(self._clean_text)
        df['sentiment'] = df['cleaned_text'].apply(self._get_sentiment)
        df['rating_sentiment_diff'] = abs(df['rating'] - df['sentiment'] * 4 + 1)
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
    
    def _build_review_graph(self, df, tfidf_matrix):
        G = nx.Graph()
        for idx, row in df.iterrows():
            G.add_node(idx, text=row['cleaned_text'], rating=row['rating'], 
                      category=row['category'], sentiment=row['sentiment'], 
                      rating_sentiment_diff=row['rating_sentiment_diff'])
        
        n_reviews = len(df)
        for i in range(n_reviews):
            for j in range(i+1, n_reviews):
                if self.text_similarity_matrix[i][j] > 0.7:
                    G.add_edge(i, j, weight=self.text_similarity_matrix[i][j], type='text_similarity')
                if df.iloc[i]['rating_sentiment_diff'] > 3 and df.iloc[j]['rating_sentiment_diff'] > 3:
                    G.add_edge(i, j, weight=0.8, type='inconsistency')
                if df.iloc[i]['category'] == df.iloc[j]['category']:
                    G.add_edge(i, j, weight=0.5, type='same_category')
        
        return G
    
    def _calculate_suspicion_scores(self, G, df):
        suspicion_scores = {}
        for node in G.nodes():
            score = 0
            edges = list(G.edges(node, data=True))
            similar_edges = [e for e in edges if e[2]['type'] == 'text_similarity']
            score += len(similar_edges) * 0.25
            inconsistent_edges = [e for e in edges if e[2]['type'] == 'inconsistency']
            score += len(inconsistent_edges) * 0.35
            category_edges = [e for e in edges if e[2]['type'] == 'same_category']
            score += len(category_edges) * 0.15
            if df.iloc[node]['rating_sentiment_diff'] > 3:
                score += 0.25
            suspicion_scores[node] = min(score, 1.0)
        return suspicion_scores
    
    def calculate_new_review_features(self, new_review_cleaned, new_rating, new_category, rating_sentiment_diff):
        if self.graph is None:
            raise ValueError("Graph not built.")
        
        features = {'suspicion_score': 0, 'similarity_to_fake': 0, 
                   'inconsistency_score': 0, 'category_anomaly': 0, 
                   'cluster_similarity': 0}
        
        fake_reviews = self.review_data[self.review_data['label'] == 1]
        if len(fake_reviews) > 0:
            similarity_scores = self._calculate_similarity_to_reviews(new_review_cleaned, fake_reviews['cleaned_text'].tolist())
            features['similarity_to_fake'] = np.max(similarity_scores) if len(similarity_scores) > 0 else 0
        
        top_suspicious = self.review_data.nlargest(10, 'graph_suspicion_score')
        if len(top_suspicious) > 0:
            similarity_to_suspicious = self._calculate_similarity_to_reviews(new_review_cleaned, top_suspicious['cleaned_text'].tolist())
            features['cluster_similarity'] = np.mean(similarity_to_suspicious) if len(similarity_to_suspicious) > 0 else 0
        
        features['inconsistency_score'] = min(rating_sentiment_diff / 4, 1.0)
        
        category_reviews = self.review_data[self.review_data['category'] == new_category]
        if len(category_reviews) > 0:
            avg_category_rating = category_reviews['rating'].mean()
            rating_diff = abs(new_rating - avg_category_rating)
            features['category_anomaly'] = min(rating_diff / 4, 1.0)
        else:
            features['category_anomaly'] = 0.3
        
        weights = {'similarity_to_fake': 0.4, 'inconsistency_score': 0.3, 
                  'cluster_similarity': 0.2, 'category_anomaly': 0.1}
        
        features['suspicion_score'] = (
            weights['similarity_to_fake'] * features['similarity_to_fake'] +
            weights['inconsistency_score'] * features['inconsistency_score'] +
            weights['cluster_similarity'] * features['cluster_similarity'] +
            weights['category_anomaly'] * features['category_anomaly']
        )
        
        return features
    
    def _calculate_similarity_to_reviews(self, new_text, existing_texts):
        if not existing_texts:
            return []
        new_vector = self.vectorizer.transform([new_text])
        existing_vectors = self.vectorizer.transform(existing_texts)
        similarities = cosine_similarity(new_vector, existing_vectors)
        return similarities[0]
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def visualize_review_graph(graph_feature_extractor):
    if graph_feature_extractor.graph is None:
        raise ValueError("Graph not built yet.")
    
    G = graph_feature_extractor.graph
    
    plt.figure(figsize=(12, 12))
    
    # Positions for all nodes
    pos = nx.spring_layout(G, seed=42)  # force-directed layout
    
    # Node colors based on suspicion score
    suspicion_scores = graph_feature_extractor.suspicion_scores
    node_colors = [suspicion_scores.get(node, 0) for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=plt.cm.Reds,
        node_size=300,
        alpha=0.8
    )
    
    # Draw edges (with different colors based on type)
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['type'] == 'text_similarity':
            edge_colors.append('blue')
        elif data['type'] == 'inconsistency':
            edge_colors.append('red')
        elif data['type'] == 'same_category':
            edge_colors.append('green')
        else:
            edge_colors.append('gray')
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5)
    
    # Draw labels (optional)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Review Graph Visualization (Node color = Suspicion score)")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                               norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, label='Suspicion Score')
    
    plt.axis('off')
    plt.show()
# from sklearn.externals import joblib
# import warnings
# warnings.filterwarnings('ignore')

# class SingleReviewPredictor:
#     def __init__(self, graph_model_path='graph_model.pkl', ml_model_path='ml_model.pkl'):
#         # Load pre-trained models
#         self.graph_extractor = GraphFeatureExtractor.load_model(graph_model_path)
        
#         # Load ML model (example - replace with your actual model)
#         try:
#             self.ml_model = joblib.load(ml_model_path)
#         except:
#             # If no ML model, use a simple rule-based fallback
#             self.ml_model = None
#             print("Warning: ML model not found. Using graph features only.")
    
#     def predict(self, review_text, rating, category):
#         """Main function to predict if a single review is fake"""
        
#         print(f"Analyzing review...")
#         print(f"Text: '{review_text[:100]}...'")
#         print(f"Rating: {rating}/5")
#         print(f"Category: {category}")
#         print("-" * 50)
        
#         # Step 1: Clean and process the new review
#         cleaned_text = self.graph_extractor._clean_text(review_text)
#         sentiment = self.graph_extractor._get_sentiment(cleaned_text)
        
#         # Convert sentiment (0-1) to 1-5 scale for comparison
#         sentiment_rating = sentiment * 4 + 1
#         rating_sentiment_diff = abs(rating - sentiment_rating)
        
#         # Step 2: Use the calculate_new_review_features function!
#         graph_features = self.graph_extractor.calculate_new_review_features(
#             new_review_cleaned=cleaned_text,
#             new_rating=rating,
#             new_category=category,
#             rating_sentiment_diff=rating_sentiment_diff
#         )
        
#         # Step 3: Get ML model prediction (if available)
#         if self.ml_model is not None:
#             ml_prediction = self.ml_model.predict_proba([review_text])[0][1]
#         else:
#             # Fallback: simple text length and exclamation marks check
#             ml_prediction = self._simple_text_analysis(review_text)
        
#         # Step 4: Combine predictions
#         final_score = self._combine_predictions(ml_prediction, graph_features['suspicion_score'])
        
#         # Step 5: Generate results
#         result = self._generate_result(final_score, ml_prediction, graph_features)
        
#         return result
    
#     def _simple_text_analysis(self, text):
#         """Simple fallback if ML model is not available"""
#         score = 0
        
#         # Too short
#         if len(text) < 20:
#             score += 0.3
        
#         # Too many exclamation marks
#         if text.count('!') > 3:
#             score += 0.2
        
#         # All caps
#         if len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.5:
#             score += 0.3
        
#         # Repeated words
#         words = text.lower().split()
#         if len(words) > 0:
#             word_counts = {}
#             for word in words:
#                 word_counts[word] = word_counts.get(word, 0) + 1
#             if max(word_counts.values()) > 3:
#                 score += 0.2
        
#         return min(score, 1.0)
    
#     def _combine_predictions(self, ml_score, graph_score, ml_weight=0.6, graph_weight=0.4):
#         """Combine ML and graph predictions"""
#         return (ml_weight * ml_score) + (graph_weight * graph_score)
    
#     def _generate_result(self, final_score, ml_score, graph_features):
#         """Generate prediction result with explanations"""
        
#         # Determine prediction
#         if final_score > 0.5:
#             prediction = "FAKE"
#             confidence = final_score * 100
#         else:
#             prediction = "REAL"
#             confidence = (1 - final_score) * 100
        
#         # Generate explanations based on features
#         explanations = []
        
#         if graph_features['similarity_to_fake'] > 0.7:
#             explanations.append(f"⚠️ Very similar to known fake reviews (similarity: {graph_features['similarity_to_fake']:.1%})")
#         elif graph_features['similarity_to_fake'] > 0.4:
#             explanations.append(f"⚠️ Somewhat similar to known fake reviews")
        
#         if graph_features['inconsistency_score'] > 0.7:
#             explanations.append(f"⚠️ Rating doesn't match review sentiment (inconsistency: {graph_features['inconsistency_score']:.1%})")
        
#         if graph_features['cluster_similarity'] > 0.6:
#             explanations.append(f"⚠️ Pattern matches suspicious review clusters")
        
#         if graph_features['category_anomaly'] > 0.6:
#             explanations.append(f"⚠️ Rating is unusual for this category")
        
#         if not explanations and final_score < 0.3:
#             explanations.append("✅ No strong indicators of fakery detected")
        
#         return {
#             'prediction': prediction,
#             'confidence': f"{confidence:.1f}%",
#             'final_score': final_score,
#             'ml_score': ml_score,
#             'graph_suspicion_score': graph_features['suspicion_score'],
#             'graph_features': graph_features,
#             'explanations': explanations
#         }
"""
Model loading and inference module.
Handles pickle file loading, text preprocessing, and predictions.
"""

import os
import pickle
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

class ModelManager:
    """Manages loading and using ML models for sentiment analysis and recommendations."""
    
    def __init__(self, pickle_dir='pickle'):
        self.pickle_dir = pickle_dir
        self.sentiment_model = None
        self.tfidf_vectorizer = None
        self.cf_recommender = None
        self.master_reviews = None
        self.models_loaded = False
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self._load_models()
    
    def _load_models(self):
        """Load all pickle files."""
        try:
            if not os.path.exists(self.pickle_dir):
                raise FileNotFoundError(f"Pickle directory not found: {self.pickle_dir}")
            
            with open(os.path.join(self.pickle_dir, 'sentiment_model.pkl'), 'rb') as f:
                self.sentiment_model = pickle.load(f)
            
            with open(os.path.join(self.pickle_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            with open(os.path.join(self.pickle_dir, 'user_based_cf.pkl'), 'rb') as f:
                self.cf_recommender = pickle.load(f)
            
            with open(os.path.join(self.pickle_dir, 'master_reviews.pkl'), 'rb') as f:
                self.master_reviews = pickle.load(f)
            
            self.models_loaded = True
            print("✓ All models loaded successfully")
        
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            self.models_loaded = False
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment prediction."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        
        return ' '.join(tokens)
    
    def predict_sentiment(self, review_text):
        """Predict sentiment for a review (1=Positive, 0=Negative)."""
        if not self.models_loaded:
            return None, None
        
        try:
            cleaned = self.preprocess_text(review_text)
            vectorized = self.tfidf_vectorizer.transform([cleaned])
            prediction = self.sentiment_model.predict(vectorized)[0]
            confidence = self.sentiment_model.predict_proba(vectorized)[0]
            
            return int(prediction), float(max(confidence))
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return None, None
    
    def get_recommendations(self, username, n_recommendations=5, sentiment_filter=True):
        """Get product recommendations for a user, optionally filtered by positive sentiment."""
        if not self.models_loaded:
            return []
        
        try:
            user_reviews = self.master_reviews[self.master_reviews['reviews_username'] == username]
            
            if len(user_reviews) == 0:
                # User not found - return popular products
                product_counts = self.master_reviews['name'].value_counts()
                return product_counts.head(n_recommendations).index.tolist()
            
            # Get CF recommendations
            cf_recs = self.cf_recommender.recommend(username, n_recommendations * 2)
            
            if sentiment_filter:
                # Filter by positive sentiment
                positive_products = []
                for product_id in cf_recs:
                    product_reviews = self.master_reviews[self.master_reviews['name'] == product_id]
                    if len(product_reviews) > 0:
                        pos_ratio = (product_reviews['sentiment'] == 1).sum() / len(product_reviews)
                        if pos_ratio >= 0.5:
                            positive_products.append(product_id)
                
                return positive_products[:n_recommendations]
            
            return cf_recs[:n_recommendations]
        
        except Exception as e:
            print(f"Error in recommendations: {e}")
            return []
    
    def sentiment_based_recommend(self, review_text, n_recommendations=5):
        """
        Analyze review sentiment and recommend similar positive products.
        Returns: sentiment (0/1), confidence, recommended products
        """
        sentiment, confidence = self.predict_sentiment(review_text)
        
        if sentiment is None:
            return None, None, []
        
        if not self.models_loaded:
            return sentiment, confidence, []
        
        try:
            if sentiment == 1:  # Positive review
                # User likes this type of product
                cleaned = self.preprocess_text(review_text)
                vectorized = self.tfidf_vectorizer.transform([cleaned])
                
                # Find similar products based on review content
                product_similarities = {}
                for product in self.master_reviews['name'].unique():
                    product_reviews = self.master_reviews[self.master_reviews['name'] == product]
                    positive_pct = (product_reviews['sentiment'] == 1).sum() / len(product_reviews)
                    product_similarities[product] = positive_pct
                
                # Sort by positive sentiment ratio and return top products
                recs = sorted(product_similarities.items(), key=lambda x: x[1], reverse=True)
                recommendations = [p[0] for p in recs[:n_recommendations]]
            else:
                # Negative review - recommend best rated products
                product_scores = {}
                for product in self.master_reviews['name'].unique():
                    product_reviews = self.master_reviews[self.master_reviews['name'] == product]
                    positive_pct = (product_reviews['sentiment'] == 1).sum() / len(product_reviews)
                    product_scores[product] = positive_pct
                
                recs = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
                recommendations = [p[0] for p in recs[:n_recommendations]]
            
            return sentiment, confidence, recommendations
        
        except Exception as e:
            print(f"Error in sentiment-based recommendations: {e}")
            return sentiment, confidence, []


# Global model manager instance
model_manager = ModelManager()

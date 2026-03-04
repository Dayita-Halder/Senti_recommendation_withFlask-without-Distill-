"""
Model loading and inference module.
Handles pickle file loading, text preprocessing, and predictions.
"""

import os
import pickle
import re
import string

# Import numpy with version check
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"CRITICAL: Failed to import numpy: {e}")
    raise

import nltk

# Set NLTK data path to a writable location (important for Railway/cloud deployments)
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)
print(f"NLTK data directory: {nltk_data_dir}")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data with better error handling
print("Downloading required NLTK data...")
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        print(f"✓ Downloaded {resource}")
    except Exception as e:
        print(f"⚠ Warning: Could not download {resource}: {e}")

# Verify NLTK data is available
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    test_stops = stopwords.words('english')
    test_lemma = WordNetLemmatizer()
    print(f"✓ NLTK data verified (loaded {len(test_stops)} stopwords)")
except Exception as e:
    print(f"⚠ NLTK data verification warning: {e}")

# ================================
# Custom Unpickler for UserBasedCF
# ================================
class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that knows about UserBasedCF."""
    
    def find_class(self, module, name):
        if name == 'UserBasedCF':
            return UserBasedCF
        return super().find_class(module, name)
class UserBasedCF:
    """
    User-based collaborative filtering recommender.
    Uses mean-centered cosine similarity.
    """
    
    def __init__(self, top_k_similar: int = 20):
        self.top_k  = top_k_similar
        self.matrix = None
        self.user_index = {}
        self.index_product = {}
        self.user_list = []
        self.raw_matrix = None

    def fit(self, rating_df):
        """Fit the model to ratings."""
        from scipy.sparse import csr_matrix
        
        user_means = rating_df.replace(0, np.nan).mean(axis=1)
        centered = rating_df.sub(user_means, axis=0).fillna(0)
        
        self.matrix = csr_matrix(centered.values)
        self.user_list = list(rating_df.index)
        self.product_list = list(rating_df.columns)
        self.user_index = {u: i for i, u in enumerate(self.user_list)}
        self.index_product = {i: p for i, p in enumerate(self.product_list)}
        self.raw_matrix = csr_matrix(rating_df.values)
        return self

    def recommend(self, username: str, n: int = 20) -> list:
        """Get top-n recommendations for a user."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        if username not in self.user_index:
            product_counts = np.asarray(self.raw_matrix.astype(bool).sum(axis=0)).flatten()
            top_idxs = np.argsort(-product_counts)[:n]
            return [self.index_product[i] for i in top_idxs]
        
        user_idx = self.user_index[username]
        user_vec = self.matrix[user_idx]
        
        sim_scores = cosine_similarity(user_vec, self.matrix).flatten()
        sim_scores[user_idx] = -1
        
        top_similar = np.argsort(-sim_scores)[:self.top_k]
        similar_sims = sim_scores[top_similar]
        
        sim_weights = similar_sims.reshape(1, -1)
        neighbor_rows = self.raw_matrix[top_similar]
        weighted_sums = sim_weights @ neighbor_rows
        weighted_sums = np.asarray(weighted_sums).flatten()
        
        already_rated = np.asarray(self.raw_matrix[user_idx].todense()).flatten() > 0
        weighted_sums[already_rated] = -np.inf
        
        top_idxs = np.argsort(-weighted_sums)[:n]
        return [self.index_product[i] for i in top_idxs if weighted_sums[i] > -np.inf]

class ModelManager:
    """Manages loading and using ML models for sentiment analysis and recommendations."""
    
    def __init__(self, pickle_dir='pickle'):
        self.pickle_dir = pickle_dir
        self.sentiment_model = None
        self.tfidf_vectorizer = None
        self.cf_recommender = None
        self.master_reviews = None
        self.models_loaded = False
        self.load_error = None  # Store error message if loading fails
        
        # Initialize NLTK components (will be set in _load_models)
        self.lemmatizer = None
        self.stop_words = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all pickle files."""
        try:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for pickle directory: {self.pickle_dir}")
            
            # Initialize NLTK components
            print("Initializing NLTK components...")
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            print("✓ NLTK components initialized")
            
            if not os.path.exists(self.pickle_dir):
                # List all files in current directory for debugging
                print(f"Contents of current directory: {os.listdir('.')}")
                raise FileNotFoundError(f"Pickle directory not found: {self.pickle_dir}")
            
            # List pickle files
            pickle_files = os.listdir(self.pickle_dir)
            print(f"Files in pickle directory: {pickle_files}")
            
            sentiment_path = os.path.join(self.pickle_dir, 'sentiment_model.pkl')
            tfidf_path = os.path.join(self.pickle_dir, 'tfidf_vectorizer.pkl')
            cf_path = os.path.join(self.pickle_dir, 'user_based_cf.pkl')
            master_path = os.path.join(self.pickle_dir, 'master_reviews.pkl')
            
            # Load sentiment model
            print("Loading sentiment model...")
            with open(sentiment_path, 'rb') as f:
                self.sentiment_model = pickle.load(f)
            
            # Load TF-IDF vectorizer
            print("Loading TF-IDF vectorizer...")
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Load CF recommender with custom unpickler
            print("Loading CF recommender...")
            with open(cf_path, 'rb') as f:
                self.cf_recommender = CustomUnpickler(f).load()
            
            # Load master reviews
            print("Loading master reviews...")
            with open(master_path, 'rb') as f:
                self.master_reviews = pickle.load(f)
            
            # Precompute product scores for faster recommendations
            print("Precomputing product scores...")
            self.product_scores = self._precompute_product_scores()
            print(f"✓ Precomputed scores for {len(self.product_scores)} products")
            
            self.models_loaded = True
            print("✓ All models loaded successfully")
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"✗ Error loading models: {error_msg}")
            import traceback
            tb_str = traceback.format_exc()
            print(tb_str)
            self.load_error = error_msg
            self.models_loaded = False
    
    @staticmethod
    def _find_class_helper(module, name):
        """Helper to find UserBasedCF class from current module."""
        if name == 'UserBasedCF':
            return UserBasedCF
        return pickle.Unpickler.find_class(module, name)
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment prediction."""
        if not self.models_loaded:
            return ""
            
        text = str(text).lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        tokens = word_tokenize(text)
        
        # Use stop_words and lemmatizer only if initialized
        if self.stop_words:
            tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        else:
            tokens = [w for w in tokens if len(w) > 2]
            
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        
        return ' '.join(tokens)
    
    def _precompute_product_scores(self):
        """Precompute product positive sentiment ratios for fast recommendations."""
        product_scores = {}
        for product in self.master_reviews['name'].unique():
            product_reviews = self.master_reviews[self.master_reviews['name'] == product]
            positive_pct = (product_reviews['sentiment_label'] == 1).sum() / len(product_reviews)
            product_scores[product] = positive_pct
        return product_scores
    
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
            # Use precomputed product scores for fast recommendations
            recs = sorted(self.product_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [p[0] for p in recs[:n_recommendations]]
            
            return sentiment, confidence, recommendations
        
        except Exception as e:
            print(f"Error in sentiment-based recommendations: {e}")
            return sentiment, confidence, []


# Global model manager instance
model_manager = ModelManager()

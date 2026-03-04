"""
Sentiment-Based Product Recommendation System — Flask API
Deployment wrapper for the trained sentiment model + collaborative filtering recommender.
"""

from flask import Flask, request, jsonify, render_template
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)

PICKLE_DIR = 'pickle'

# ================================
# Text Preprocessing & Imports
# ================================
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ================================
# UserBasedCF Class Definition
# ================================
class UserBasedCF:
    """
    Memory-efficient user-based collaborative filter.
    - Uses mean-centered cosine similarity to normalise per-user rating scale
    - Recommends products not yet rated by the target user
    """

    def __init__(self, top_k_similar: int = 20):
        self.top_k  = top_k_similar
        self.matrix = None
        self.user_index    = {}
        self.index_product = {}
        self.product_index = {}
        self.user_list     = []

    def fit(self, rating_df):
        """
        Args:
            rating_df: user × product matrix (rows=users, cols=products, vals=ratings)
        """
        from scipy.sparse import csr_matrix
        
        # Mean-center each user's ratings (removes user-level bias)
        user_means = rating_df.replace(0, np.nan).mean(axis=1)
        centered   = rating_df.sub(user_means, axis=0).fillna(0)

        self.matrix        = csr_matrix(centered.values)
        self.user_list     = list(rating_df.index)
        self.product_list  = list(rating_df.columns)
        self.user_index    = {u: i for i, u in enumerate(self.user_list)}
        self.index_product = {i: p for i, p in enumerate(self.product_list)}
        self.raw_matrix    = csr_matrix(rating_df.values)
        return self

    def recommend(self, username: str, n: int = 20) -> list:
        """
        Returns top-n unrated product names for the given user.
        Falls back to popularity-based ranking if user not found.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if username not in self.user_index:
            # Cold-start: return most-reviewed products
            product_counts = np.asarray(self.raw_matrix.astype(bool).sum(axis=0)).flatten()
            top_idxs       = np.argsort(-product_counts)[:n]
            return [self.index_product[i] for i in top_idxs]

        user_idx  = self.user_index[username]
        user_vec  = self.matrix[user_idx]

        # Compute similarity to all other users
        sim_scores  = cosine_similarity(user_vec, self.matrix).flatten()
        sim_scores[user_idx] = -1  # exclude self

        # Top-k most similar users
        top_similar = np.argsort(-sim_scores)[:self.top_k]
        similar_sims = sim_scores[top_similar]

        # Weighted sum of similar users' ratings
        sim_weights   = similar_sims.reshape(1, -1)
        neighbor_rows = self.raw_matrix[top_similar]  # shape: (k, products)
        weighted_sums = sim_weights @ neighbor_rows    # shape: (1, products)
        weighted_sums = np.asarray(weighted_sums).flatten()

        # Mask already-rated products
        already_rated = np.asarray(self.raw_matrix[user_idx].todense()).flatten() > 0
        weighted_sums[already_rated] = -np.inf

        top_idxs = np.argsort(-weighted_sums)[:n]
        return [self.index_product[i] for i in top_idxs if weighted_sums[i] > -np.inf]

# ================================
# Custom Pickle Loader with Module Remapping
# ================================
class ModuleRemappingUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps __main__ to current module."""
    
    def find_class(self, module, name):
        # Remap __main__ references to current module
        if module == '__main__':
            # Get the current module where this unpickler is defined
            current_module = sys.modules['__main__']
            if hasattr(current_module, name):
                return getattr(current_module, name)
        return super().find_class(module, name)

def load_pickle(fname):
    """Helper function to load pickle files with module remapping."""
    with open(os.path.join(PICKLE_DIR, fname), 'rb') as f:
        return ModuleRemappingUnpickler(f).load()

# ================================
# Load Artefacts at Startup
# ================================
print('Loading deployment artefacts...')
try:
    # Check if pickle directory exists
    if not os.path.exists(PICKLE_DIR):
        print(f'❌ Pickle directory not found at {PICKLE_DIR}')
        print(f'Current directory: {os.getcwd()}')
        print(f'Directory contents: {os.listdir(".")}')
        raise FileNotFoundError(f'Pickle directory "{PICKLE_DIR}" not found')
    
    sentiment_model = load_pickle('sentiment_model.pkl')
    tfidf_vectorizer = load_pickle('tfidf_vectorizer.pkl')
    cf_recommender = load_pickle('user_based_cf.pkl')
    master_reviews = load_pickle('master_reviews.pkl')
    print('✔ All artefacts loaded successfully.')
except Exception as e:
    print(f'❌ Error loading artefacts: {e}')
    print('Make sure to run the notebook first to generate pickle files.')
    # Don't exit - let the app start but return error for API calls
    sentiment_model = None
    tfidf_vectorizer = None
    cf_recommender = None
    master_reviews = None

def preprocess_text(text):
    """
    Clean and preprocess text for sentiment analysis.
    Steps: lowercase, remove URLs/emails/mentions/numbers, remove punctuation,
           tokenize, remove stopwords, lemmatize.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# ================================
# Sentiment Filtering Function
# ================================
def sentiment_filter(candidate_products, top_n=5):
    """
    Score candidate products by positive sentiment ratio.
    
    Args:
        candidate_products: List of product names from CF model
        top_n: Number of final recommendations
        
    Returns:
        List of dictionaries with product details and sentiment scores
    """
    product_scores = []
    
    for product_name in candidate_products:
        product_reviews = master_reviews[master_reviews['name'] == product_name]['processed_text']
        
        if len(product_reviews) == 0:
            continue
        
        # Transform reviews using TF-IDF
        features = tfidf_vectorizer.transform(product_reviews)
        
        # Predict sentiment
        predictions = sentiment_model.predict(features)
        positive_count = int(predictions.sum())
        total = len(predictions)
        
        product_scores.append({
            'product': product_name,
            'positive_count': positive_count,
            'total_reviews': total,
            'positive_ratio': round(positive_count / total, 4)
        })
    
    # Sort by positive ratio and return top_n
    product_scores.sort(key=lambda x: x['positive_ratio'], reverse=True)
    return product_scores[:top_n]

# ================================
# Flask Routes
# ================================

@app.route('/')
def home():
    """Render the home page with input form."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for getting recommendations.
    
    Expected JSON input:
    {
        "username": "john_doe",
        "num_candidates": 20,
        "num_recommendations": 5
    }
    
    Returns JSON response with recommendations.
    """
    try:
        if master_reviews is None:
            return jsonify({'error': 'Models not loaded. Pickle files missing.'}), 503
        
        data = request.get_json()
        
        if not data or 'username' not in data:
            return jsonify({'error': 'Missing username parameter'}), 400
        
        username = data['username']
        num_candidates = data.get('num_candidates', 20)
        num_recommendations = data.get('num_recommendations', 5)
        
        # Check if user exists
        if username not in master_reviews['reviews_username'].values:
            return jsonify({'error': f'User "{username}" not found in the system'}), 404
        
        # Step 1: Get candidate products from collaborative filtering
        try:
            candidates = cf_recommender.recommend(username, n=num_candidates)
        except Exception as e:
            return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500
        
        if not candidates:
            return jsonify({'error': 'No recommendations available for this user'}), 404
        
        # Step 2: Apply sentiment filtering
        final_recommendations = sentiment_filter(candidates, top_n=num_recommendations)
        
        return jsonify({
            'username': username,
            'recommendations': final_recommendations,
            'total_candidates': len(candidates)
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    API endpoint for analyzing sentiment of a single review.
    
    Expected JSON input:
    {
        "text": "This product is amazing!"
    }
    
    Returns JSON response with sentiment prediction.
    """
    try:
        if sentiment_model is None or tfidf_vectorizer is None:
            return jsonify({'error': 'Sentiment model not loaded'}), 503
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Transform using TF-IDF
        features = tfidf_vectorizer.transform([processed_text])
        
        # Predict sentiment
        prediction = sentiment_model.predict(features)[0]
        probability = sentiment_model.predict_proba(features)[0]
        
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        confidence = float(max(probability))
        
        return jsonify({
            'original_text': text,
            'processed_text': processed_text,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'negative': round(float(probability[0]), 4),
                'positive': round(float(probability[1]), 4)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """
    API endpoint to get list of available users.
    Returns a sample of users in the system.
    """
    try:
        if master_reviews is None:
            return jsonify({'error': 'Data not loaded'}), 503
        
        # Get users with most reviews
        user_counts = master_reviews['reviews_username'].value_counts().head(50)
        users = [{'username': user, 'review_count': int(count)} 
                 for user, count in user_counts.items()]
        
        return jsonify({
            'total_users': len(master_reviews['reviews_username'].unique()),
            'sample_users': users
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """
    API endpoint to get list of available products.
    Returns a sample of products in the system.
    """
    try:
        if master_reviews is None:
            return jsonify({'error': 'Data not loaded'}), 503
        
        # Get products with most reviews
        product_counts = master_reviews['name'].value_counts().head(50)
        products = [{'product_name': product, 'review_count': int(count)} 
                    for product, count in product_counts.items()]
        
        return jsonify({
            'total_products': len(master_reviews['name'].unique()),
            'sample_products': products
        })
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if master_reviews is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Pickle files not loaded',
            'models_loaded': False
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'total_reviews': len(master_reviews),
        'total_users': len(master_reviews['reviews_username'].unique()),
        'total_products': len(master_reviews['name'].unique())
    })

# ================================
# Run Flask App
# ================================
if __name__ == '__main__':
    print('\n' + '='*60)
    print('🚀 Sentiment-Based Recommendation System — Flask API')
    print('='*60)
    print(f'📊 Loaded {len(master_reviews)} reviews')
    print(f'👥 Users: {len(master_reviews["reviews_username"].unique())}')
    print(f'📦 Products: {len(master_reviews["name"].unique())}')
    print('='*60 + '\n')
    
    app.run(debug=True, host='0.0.0.0', port=5000)

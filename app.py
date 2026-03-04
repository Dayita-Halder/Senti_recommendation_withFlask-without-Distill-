"""
Sentiment-Based Product Recommendation System - Flask API
Combines sentiment analysis with product recommendations.
Route: / -> sentiment-based recommendations from review text
"""

from flask import Flask, render_template, request, jsonify
from model import model_manager

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')

@app.route('/api/echo', methods=['POST'])
def echo():
    """Echo test endpoint."""
    data = request.json
    return jsonify({"echo": data.get('review', 'no data')}), 200

@app.route('/api/sentiment-recommend', methods=['POST'])
def sentiment_recommend():
    """
    Analyze review sentiment and provide product recommendations.
    Simpler version that avoids sklearn issues.
    """
    try:
        import time
        start_time = time.time()
        
        data = request.json
        review_text = data.get('review', '').strip()
        
        if not review_text:
            return jsonify({"error": "Review text is required"}), 400
        
        if not model_manager.models_loaded:
            return jsonify({"error": "Models not loaded"}), 503
        
        # Simple sentiment: check for positive/negative keywords
        positive_words = {'good', 'great', 'amazing', 'excellent', 'wonderful', 'fantastic', 'love', 'awesome', 'perfect', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor', 'disappointing'}
        
        review_lower = review_text.lower()
        pos_count = sum(1 for word in positive_words if word in review_lower)
        neg_count = sum(1 for word in negative_words if word in review_lower)
        
        # Fallback to ML if keywords don't help
        if pos_count == 0 and neg_count == 0:
            try:
                sentiment, confidence = model_manager.predict_sentiment(review_text)
                if sentiment is None:
                    sentiment = 1  # Default to positive
                    confidence = 0.5
            except:
                sentiment = 1
                confidence = 0.5
        else:
            sentiment = 1 if pos_count > neg_count else 0
            confidence = 0.8
        
        n_recs = data.get('n_recommendations', 5)
        n_recs = max(1, min(int(n_recs), 10))
        
        # Get recommendations
        try:
            recs = sorted(model_manager.product_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [p[0] for p in recs[:n_recs]]
        except:
            recommendations = []
        
        elapsed = time.time() - start_time
        print(f"Processed request in {elapsed:.2f}s")
        
        return jsonify({
            "sentiment": int(sentiment),
            "sentiment_label": "Positive ✓" if sentiment == 1 else "Negative ✗",
            "confidence": round(confidence * 100, 2),
            "recommendations": recommendations,
            "review_preview": (review_text[:80] + "...") if len(review_text) > 80 else review_text
        })
    
    except Exception as e:
        import traceback
        print(f"Error in sentiment-recommend: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Check system health and model status."""
    if not model_manager.models_loaded:
        return jsonify({
            "status": "error",
            "models_loaded": False,
            "message": "Pickle files not loaded",
            "error": model_manager.load_error or "Unknown error during model loading",
            "hint": "Check pickle/ directory exists with all required .pkl files"
        }), 503
    
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "total_reviews": len(model_manager.master_reviews),
        "total_users": len(model_manager.master_reviews['reviews_username'].unique()),
        "total_products": len(model_manager.master_reviews['name'].unique())
    })

@app.route('/api/debug', methods=['GET'])
def debug():
    """Debug endpoint to check file system."""
    import os
    import sys
    
    debug_info = {
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "pickle_dir_exists": os.path.exists('pickle'),
        "files_in_cwd": os.listdir('.'),
    }
    
    # Check numpy version
    try:
        import numpy as np
        debug_info['numpy_version'] = np.__version__
        debug_info['numpy_core_exists'] = hasattr(np, '_core')
    except Exception as e:
        debug_info['numpy_error'] = str(e)
    
    if os.path.exists('pickle'):
        debug_info['files_in_pickle'] = os.listdir('pickle')
        debug_info['pickle_file_sizes'] = {}
        for f in os.listdir('pickle'):
            try:
                size = os.path.getsize(os.path.join('pickle', f))
                debug_info['pickle_file_sizes'][f] = f"{size / (1024*1024):.2f} MB"
            except:
                pass
    else:
        debug_info['files_in_pickle'] = []
    
    return jsonify(debug_info)

@app.route('/api/test-sentiment', methods=['GET'])
def test_sentiment():
    """Test endpoint to debug sentiment prediction."""
    import time
    
    test_review = "This is an amazing product! Love it!"
    start = time.time()
    
    try:
        sent, conf, recs = model_manager.sentiment_based_recommend(test_review, 5)
        elapsed = time.time() - start
        
        return jsonify({
            "status": "success",
            "sentiment": int(sent) if sent is not None else None,
            "confidence": round(conf * 100, 2) if conf else None,
            "recommendations_count": len(recs),
            "sample_recs": recs[:3],
            "elapsed_seconds": round(elapsed, 3)
        })
    except Exception as e:
        elapsed = time.time() - start
        import traceback
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(elapsed, 3)
        }), 500

if __name__ == '__main__':
    print('\n' + '='*70)
    print('🚀 Sentiment-Based Recommendation System')
    print('='*70)
    print(f'Models Loaded: {model_manager.models_loaded}')
    if model_manager.models_loaded:
        print(f'Reviews: {len(model_manager.master_reviews):,}')
        print(f'Users: {len(model_manager.master_reviews["reviews_username"].unique()):,}')
        print(f'Products: {len(model_manager.master_reviews["name"].unique()):,}')
    else:
        print('⚠️  Models failed to load. Check pickle/ directory.')
    print('='*70)
    print('Starting Flask app on http://localhost:5000')
    print('='*70 + '\n')
    
    app.run(debug=False, host='0.0.0.0', port=5000)

"""
Serverless handler for Vercel.
Routes all requests to the Flask application.
"""

import sys
import os
import traceback

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try to import Flask app, with detailed error reporting
try:
    from app import app
    print("✓ Flask app imported successfully")
except Exception as e:
    print(f"✗ Failed to import Flask app: {e}")
    traceback.print_exc()
    # Create a minimal error app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return jsonify({"error": f"Failed to load main app: {str(e)}", "details": traceback.format_exc()}), 500

# Export as handler for Vercel
handler = app

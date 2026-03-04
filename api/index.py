"""
Serverless handler for Vercel.
Routes all requests to the Flask application.
"""

import sys
import os

# Add project root to sys.path so we can import app.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Flask app
try:
    from app import app
except ImportError as e:
    print(f"Failed to import app: {e}")
    raise

# Export app as WSGI application for Vercel
application = app

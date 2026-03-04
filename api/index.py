"""
Vercel deployment wrapper for Flask app.
Exposes the Flask app for Vercel's serverless environment.
"""

import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export the Flask app for Vercel
def handler(request):
    return app(request)

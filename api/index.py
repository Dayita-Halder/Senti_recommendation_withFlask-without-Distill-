"""
Flask app entry point for Vercel serverless deployment.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import Flask app
from app import app

# For Vercel serverless
if __name__ == "__main__":
    app.run()

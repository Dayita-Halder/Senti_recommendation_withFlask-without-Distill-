"""
Quick setup verification script for the Flask app.
Checks if all required files and dependencies are in place.
"""

import os
import sys

def check_file(filepath, required=True):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {filepath}")
    return exists

def check_directory(dirpath):
    """Check if a directory exists."""
    exists = os.path.exists(dirpath)
    status = "✓" if exists else "✗"
    print(f"  {status} {dirpath}/")
    return exists

def main():
    print("=" * 60)
    print("Flask App Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check Python packages
    print("\n1. Checking Python packages...")
    required_packages = [
        'flask', 'numpy', 'pandas', 'sklearn', 'scipy', 'nltk'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (run: pip install -r requirements.txt)")
            all_good = False
    
    # Check required files
    print("\n2. Checking required files...")
    files_exist = {
        'app.py': check_file('app.py'),
        'requirements.txt': check_file('requirements.txt'),
        'templates/index.html': check_file('templates/index.html'),
        'sample30.csv': check_file('sample30.csv', required=False)
    }
    
    # Check pickle directory and files
    print("\n3. Checking pickle directory and model files...")
    pickle_dir_exists = check_directory('pickle')
    
    if pickle_dir_exists:
        pickle_files = [
            'pickle/sentiment_model.pkl',
            'pickle/tfidf_vectorizer.pkl',  
            'pickle/user_based_cf.pkl',
            'pickle/master_reviews.pkl'
        ]
        
        for pfile in pickle_files:
            if not check_file(pfile):
                all_good = False
    else:
        print("  ⚠ Pickle directory not found. Run notebook to generate models.")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good and files_exist['sample30.csv']:
        print("✓ All checks passed! You can run the Flask app.")
        print("\nTo start the app, run:")
        print("  python app.py")
    elif not files_exist['sample30.csv']:
        print("⚠ Data file missing!")
        print("\nNext steps:")
        print("  1. Place 'sample30.csv' in this directory")
        print("  2. Open and run all cells in the Jupyter notebook")
        print("  3. This will generate the pickle/ directory with model files")
        print("  4. Then run: python app.py")
    elif not pickle_dir_exists:
        print("⚠ Model files not generated yet!")
        print("\nNext steps:")
        print("  1. Open 'sentiment_recommendation_notebook(1).ipynb' in Jupyter")
        print("  2. Run all cells (Cell → Run All)")
        print("  3. Wait for completion (creates pickle/ folder)")
        print("  4. Then run: python app.py")
    else:
        print("✗ Some requirements are missing. Check above.")
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == '__main__':
    main()

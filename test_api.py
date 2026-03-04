"""
Test script for the Flask API endpoints.
Run this after starting the Flask app to verify all endpoints work correctly.
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health check endpoint."""
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_users():
    """Test the get users endpoint."""
    print("\n2. Testing Get Users...")
    try:
        response = requests.get(f"{BASE_URL}/api/users")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Total users: {data.get('total_users')}")
        if data.get('sample_users'):
            print(f"   Sample user: {data['sample_users'][0]}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_products():
    """Test the get products endpoint."""
    print("\n3. Testing Get Products...")
    try:
        response = requests.get(f"{BASE_URL}/api/products")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Total products: {data.get('total_products')}")
        if data.get('sample_products'):
            print(f"   Sample product: {data['sample_products'][0]}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_sentiment():
    """Test the sentiment analysis endpoint."""
    print("\n4. Testing Sentiment Analysis...")
    test_reviews = [
        "This product is absolutely amazing! Works perfectly and exceeded my expectations.",
        "Terrible product. Broke after one day. Complete waste of money."
    ]
    
    try:
        for review in test_reviews:
            response = requests.post(
                f"{BASE_URL}/api/sentiment",
                json={"text": review},
                headers={"Content-Type": "application/json"}
            )
            print(f"   Status: {response.status_code}")
            data = response.json()
            print(f"   Review: {review[:50]}...")
            print(f"   Sentiment: {data.get('sentiment')} ({data.get('confidence', 0)*100:.1f}% confidence)")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_recommend(username=None):
    """Test the recommendation endpoint."""
    print("\n5. Testing Recommendations...")
    
    # First, get a valid username if not provided
    if not username:
        try:
            response = requests.get(f"{BASE_URL}/api/users")
            data = response.json()
            if data.get('sample_users'):
                username = data['sample_users'][0]['username']
                print(f"   Using username: {username}")
        except:
            print("   Error: Could not fetch sample username")
            return False
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/recommend",
            json={
                "username": username,
                "num_candidates": 20,
                "num_recommendations": 5
            },
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Candidates analyzed: {data.get('total_candidates')}")
            print(f"   Recommendations returned: {len(data.get('recommendations', []))}")
            
            if data.get('recommendations'):
                print("\n   Top recommendation:")
                rec = data['recommendations'][0]
                print(f"   - Product: {rec['product']}")
                print(f"   - Positive ratio: {rec['positive_ratio']*100:.1f}%")
                print(f"   - Reviews: {rec['positive_count']}/{rec['total_reviews']} positive")
        else:
            print(f"   Error: {response.json()}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def main():
    """Run all API tests."""
    print("=" * 60)
    print("Flask API Testing Suite")
    print("=" * 60)
    print("\nMake sure the Flask app is running on port 5000!")
    print("Start it with: python app.py")
    
    input("\nPress Enter to begin testing...")
    
    results = {
        "Health Check": test_health(),
        "Get Users": test_users(),
        "Get Products": test_products(),
        "Sentiment Analysis": test_sentiment(),
        "Recommendations": test_recommend()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: 'requests' package not found.")
        print("Install it with: pip install requests")
        exit(1)
    
    main()

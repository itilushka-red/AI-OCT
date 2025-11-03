"""
Simple script to test the OCT API
"""

import requests

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predict(image_path):
    """Test prediction endpoint"""
    print(f"Testing /predict endpoint with: {image_path}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"\nAll Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob}%")
        print(f"\nInference Time: {result['inference_time_ms']} ms")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    # Test health
    test_health()

    # Test prediction - UPDATE THIS PATH to your test image
    test_image = "C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/test/DME/DME-11053-1.jpeg"
    test_predict(test_image)

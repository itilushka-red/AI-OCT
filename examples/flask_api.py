#!/usr/bin/env python3
"""
Flask REST API Example - OCT Classification

Simple REST API for OCT image classification.

Installation:
    pip install flask

Usage:
    python flask_api.py

    # Then test with:
    curl -X POST -F "file=@scan.jpg" http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from oct_predictor import OCTPredictor, validate_image
import os
import tempfile

app = Flask(__name__)

# Initialize predictor once (not on every request)
print("Loading model...")
predictor = OCTPredictor('models/best_oct_model.h5')
print("✓ Model loaded - API ready!")

@app.route('/')
def home():
    """API information endpoint"""
    return jsonify({
        'name': 'OCT Classification API',
        'version': '1.0',
        'endpoints': {
            '/': 'GET - API information',
            '/predict': 'POST - Predict OCT image class',
            '/health': 'GET - Health check'
        },
        'usage': 'curl -X POST -F "file=@scan.jpg" http://localhost:5000/predict'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict OCT image classification

    Expected: multipart/form-data with 'file' field
    Returns: JSON with prediction results
    """
    # Check if file was provided
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided. Use: curl -X POST -F "file=@image.jpg" ...'
        }), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        # Validate image
        is_valid, error_msg = validate_image(temp_path)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400

        # Make prediction
        result = predictor.predict(temp_path)
        result['success'] = True
        result['filename'] = file.filename

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("OCT Classification REST API")
    print("=" * 60)
    print("\nAPI running at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Predict image")
    print("\nTest with:")
    print("  curl -X POST -F \"file=@scan.jpg\" http://localhost:5000/predict")
    print("\n" + "=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)

"""
OCT Image Predictor Module

A reusable module for OCT (Optical Coherence Tomography) image classification.
Can be imported in other Python applications.

Usage:
    from oct_predictor import OCTPredictor

    predictor = OCTPredictor('models/best_oct_model.h5')
    result = predictor.predict('path/to/image.jpg')
    print(result)
"""

import os
import json
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


class OCTPredictor:
    """
    OCT Image Classification Predictor

    Loads a trained model and provides prediction functionality
    for retinal OCT images.
    """

    # Class names for OCT conditions
    CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    # Full names for display
    CLASS_FULL_NAMES = {
        'CNV': 'Choroidal Neovascularization',
        'DME': 'Diabetic Macular Edema',
        'DRUSEN': 'Drusen',
        'NORMAL': 'Normal (Healthy)'
    }

    def __init__(self, model_path, img_size=224):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path (str): Path to the trained model file (.h5)
            img_size (int): Input image size (default: 224 for EfficientNetB0)

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please ensure the model has been trained and saved."
            )

        try:
            print(f"Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def _preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.

        Args:
            image_path (str): Path to the image file

        Returns:
            np.ndarray: Preprocessed image array

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image processing fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load image
            img = Image.open(image_path)

            # Convert to RGB (in case image is grayscale or has alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to model input size
            img = img.resize((self.img_size, self.img_size))

            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")

    def predict(self, image_path, return_probabilities=True):
        """
        Predict the class of an OCT image.

        Args:
            image_path (str): Path to the OCT image
            return_probabilities (bool): Include all class probabilities

        Returns:
            dict: Prediction results containing:
                - predicted_class (str): Predicted class name
                - confidence (float): Confidence percentage (0-100)
                - probabilities (dict): Probabilities for all classes
                - full_name (str): Full medical name of condition
                - inference_time_ms (float): Inference time in milliseconds

        Raises:
            FileNotFoundError: If image doesn't exist
            Exception: If prediction fails
        """
        start_time = time.time()

        try:
            # Preprocess image
            img_array = self._preprocess_image(image_path)

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)

            # Get predicted class
            predicted_idx = np.argmax(predictions[0])
            predicted_class = self.CLASS_NAMES[predicted_idx]
            confidence = float(predictions[0][predicted_idx]) * 100

            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Build result dictionary
            result = {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'full_name': self.CLASS_FULL_NAMES[predicted_class],
                'inference_time_ms': round(inference_time, 2)
            }

            # Add probabilities if requested
            if return_probabilities:
                probabilities = {}
                for i, class_name in enumerate(self.CLASS_NAMES):
                    probabilities[class_name] = round(float(predictions[0][i]) * 100, 2)
                result['probabilities'] = probabilities

            return result

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.

        Args:
            image_paths (list): List of image file paths

        Returns:
            list: List of prediction results (one per image)
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                result['success'] = True
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        return results

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            dict: Model information
        """
        if self.model is None:
            return None

        return {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': self.model.count_params(),
            'classes': self.CLASS_NAMES
        }


def validate_image(image_path):
    """
    Validate that a file is a valid image.

    Args:
        image_path (str): Path to image file

    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(image_path):
        return False, f"File not found: {image_path}"

    # Check if it's a file (not a directory)
    if not os.path.isfile(image_path):
        return False, f"Path is not a file: {image_path}"

    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        return False, f"Invalid file format: {ext}. Supported: {', '.join(valid_extensions)}"

    # Try to open with PIL
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid or corrupted image: {str(e)}"


if __name__ == "__main__":
    # Example usage when running this module directly
    print("OCT Predictor Module")
    print("=" * 50)
    print("\nThis is a module file. Import it in your Python code:")
    print("\n  from oct_predictor import OCTPredictor")
    print("  predictor = OCTPredictor('models/best_oct_model.h5')")
    print("  result = predictor.predict('image.jpg')")
    print("\nOr use the CLI script: python predict.py image.jpg")

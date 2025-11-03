"""
FastAPI Backend for OCT Image Classification
Simple API to receive images and return disease predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCT Disease Classification API",
    description="API for classifying retinal OCT images",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
MODEL_PATH = "models/simple_oct_final.h5"
IMG_SIZE = 224
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load the trained model when the API starts"""
    global MODEL
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        MODEL = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model: {e}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for model inference"""
    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OCT Disease Classification API",
        "status": "running",
        "model_loaded": MODEL is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an OCT image and get disease classification prediction

    Returns:
        - predicted_class: The predicted disease class
        - confidence: Confidence percentage for the prediction
        - probabilities: All class probabilities
        - inference_time_ms: Time taken for inference
    """
    # Check if model is loaded
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type (if provided)
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        # Read image bytes
        start_time = time.time()
        image_bytes = await file.read()

        # Check if file is empty
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Preprocess image
        try:
            img_array = preprocess_image(image_bytes)
        except Exception as img_error:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file. Could not process image: {str(img_error)}"
            )

        # Make prediction
        predictions = MODEL.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx]) * 100

        # Get all class probabilities
        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"Prediction: {predicted_class} ({confidence:.2f}%) for {file.filename}")

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "inference_time_ms": round(inference_time, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("="*70)
    print("Starting OCT Classification API Server")
    print("="*70)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("="*70)

    uvicorn.run(app, host="0.0.0.0", port=8000)

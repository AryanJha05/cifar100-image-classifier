"""
CIFAR-100 CNN Model Predictor Module
Centralized model loading, image preprocessing, and inference execution.
Strictly synchronized with the trained ML notebook architecture (EfficientNetV2B0, 96x96 RGB).
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

# ML Model Contract Constants
IMG_SIZE = 96
NUM_CLASSES = 100

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"

_model = None
_class_names: List[str] = []


def load_class_names() -> List[str]:
    """Load the CIFAR-100 class names mapping."""
    global _class_names
    if not _class_names:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _class_names = json.load(f)
        else:
            raise FileNotFoundError(
                f"Class names mapping file not found at {CLASS_NAMES_PATH}"
            )
    return _class_names


def get_model():
    """Lazily load and cache the trained Keras model."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model artifact 'best_model.keras' not found at {MODEL_PATH}. "
                "Please export and place your trained model from the notebook into backend/model/."
            )
        import tensorflow as tf

        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image bytes to match notebook inference pipeline.

    - Convert to RGB (3 channels)
    - Resize to 96x96 spatial dimensions
    - Convert to NumPy array
    - Expand dimensions to (1, 96, 96, 3) batch format
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)
    return input_tensor


def predict_image(image_bytes: bytes) -> Dict[str, Any]:
    """Run feedforward inference on uploaded image and return top-5 predictions."""
    class_names = load_class_names()
    input_tensor = preprocess_image(image_bytes)

    try:
        model = get_model()
        t0 = time.perf_counter()
        prediction = model.predict(input_tensor, verbose=0)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {str(e)}")

    probabilities = prediction[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx] * 100)

    top_5_indices = np.argsort(probabilities)[-5:][::-1]

    top_predictions = [
        {
            "class_name": str(class_names[idx]),
            "confidence": round(float(probabilities[idx] * 100), 2),
        }
        for idx in top_5_indices
    ]

    return {
        "predicted_class": str(class_names[predicted_idx]),
        "confidence": round(confidence, 2),
        "top_predictions": top_predictions,
        "top_5": top_predictions,
        "model": "EfficientNetV2B0",
        "classes": NUM_CLASSES,
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB",
        "inference_time_ms": inference_time_ms,
    }

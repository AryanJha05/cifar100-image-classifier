"""
CIFAR-100 CNN Model Predictor Module
Centralized model loading, image preprocessing, dual-output feature extraction,
and Out-of-Distribution (OOD) / invalid image rejection.
Strictly synchronized with the trained ML notebook architecture (EfficientNetV2B0, 96x96 RGB).
"""

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from backend import ood_config
    from backend.ood_detector import get_ood_detector
except ImportError:
    import ood_config
    from ood_detector import get_ood_detector

logger = logging.getLogger("cifar100.predictor")

# ML Model Contract Constants
IMG_SIZE = 96
NUM_CLASSES = 100

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"

_inference_model = None
_class_names: List[str] = []


def load_class_names() -> List[str]:
    """Load and cache the 100 CIFAR-100 class labels."""
    global _class_names
    if not _class_names:
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            _class_names = json.load(f)
        if len(_class_names) != NUM_CLASSES:
            logger.warning(
                f"Expected {NUM_CLASSES} class names, but loaded {len(_class_names)}"
            )
    return _class_names


def get_inference_model():
    """
    Lazily load and cache the trained Keras model with multi-scale outputs.
    Outputs:
      1. Penultimate feature embedding from `dense_512` (shape: [batch, 512])
      2. Convolutional backbone embedding from `avg_pool` (shape: [batch, 1280])
      3. Softmax class probability distribution (shape: [batch, 100])
    """
    global _inference_model
    if _inference_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model artifact 'best_model.keras' not found at {MODEL_PATH}. "
                "Please export and place your trained model from the notebook into backend/model/."
            )
        import tensorflow as tf

        # Configure memory growth for GPU if available
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass

        full_model = tf.keras.models.load_model(str(MODEL_PATH))
        dense_layer = full_model.get_layer("dense_512")
        pool_layer = full_model.get_layer("avg_pool")
        _inference_model = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=[dense_layer.output, pool_layer.output, full_model.output],
            name="multi_scale_cifar100_inference"
        )
        logger.info(
            f"Loaded multi-scale inference model with dense layer '{dense_layer.name}' "
            f"and backbone layer '{pool_layer.name}'"
        )
    return _inference_model


def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocess uploaded image bytes to match notebook inference pipeline.
    - Validate image data via PIL
    - Convert to RGB (3 channels)
    - Resize to 96x96 spatial dimensions
    - Convert to NumPy array
    - Expand dimensions to (1, 96, 96, 3) batch format
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not decode image file: {str(e)}")

    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32)
    input_tensor = np.expand_dims(img_array, axis=0)
    return input_tensor, pil_img


def predict_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Execute end-to-end inference and OOD verification on uploaded image.

    Flow:
    1. Input validation & spatial variance check (filters flat/blank/corrupt inputs).
    2. Feedforward pass through dual-output EfficientNetV2B0 model.
    3. Extract 512-d feature representation & 100-class softmax probabilities.
    4. Dual-layer OOD check:
       - Feature-space prototype cosine similarity
       - Classification confidence
    5. Returns structured JSON:
       - If valid: { "valid": true, "predicted_class": ..., "confidence": ..., "top_5": [...] }
       - If rejected: { "valid": false, "predicted_class": null, "confidence": null, "top_5": [], "message": ... }
    """
    class_names = load_class_names()
    input_tensor, pil_img = preprocess_image(image_bytes)

    # 1. Natural Photographic Image Domain & Degeneracy Safeguard:
    # Reject flat uniform colors, digital UI, app icons, vector logos, and text documents
    detector = get_ood_detector()
    domain_ok, domain_reason = detector.evaluate_image_domain(pil_img)
    if not domain_ok:
        logger.info(f"OOD Rejected at domain stage: {domain_reason}")
        return {
            "valid": False,
            "predicted_class": None,
            "confidence": None,
            "top_predictions": [],
            "top_5": [],
            "message": ood_config.REJECTION_MESSAGE,
            "model": "EfficientNetV2B0",
            "classes": NUM_CLASSES,
            "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB",
            "inference_time_ms": 0.5,
            "ood_metrics": {
                "max_similarity": 0.0,
                "decision": "rejected",
                "reasons": [domain_reason]
            }
        }

    # 2. Multi-Scale Forward Pass
    try:
        model = get_inference_model()
        t0 = time.perf_counter()
        dense_embs, pool_embs, probabilities_batch = model(input_tensor, training=False)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        dense_emb = dense_embs.numpy()[0]
        backbone_emb = pool_embs.numpy()[0]
        probabilities = probabilities_batch.numpy()[0]
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {str(e)}")

    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx] * 100)

    # 3. Multi-Scale OOD & Domain Boundary Evaluation
    ood_result = detector.evaluate(
        dense_embedding=dense_emb,
        confidence=confidence,
        backbone_embedding=backbone_emb,
        pil_img=pil_img
    )

    # Top-5 rankings
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_predictions = [
        {
            "class_name": str(class_names[idx]),
            "confidence": round(float(probabilities[idx] * 100), 2),
        }
        for idx in top_5_indices
    ]

    # 4. Final Decision Routing
    if not ood_result["is_valid"]:
        logger.info(
            f"OOD Rejected: forced={class_names[predicted_idx]} ({confidence:.1f}%), "
            f"dense_sim={ood_result['max_similarity']:.3f}, backbone_sim={ood_result.get('backbone_similarity', 0.0):.3f}, "
            f"reasons={ood_result['reasons']}"
        )
        return {
            "valid": False,
            "predicted_class": None,
            "confidence": None,
            "top_predictions": [],
            "top_5": [],
            "message": ood_config.REJECTION_MESSAGE,
            "model": "EfficientNetV2B0",
            "classes": NUM_CLASSES,
            "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB",
            "inference_time_ms": inference_time_ms,
            "ood_metrics": {
                "max_similarity": ood_result["max_similarity"],
                "backbone_similarity": ood_result.get("backbone_similarity"),
                "nearest_prototype_class": str(class_names[ood_result["nearest_proto_idx"]]),
                "decision": "rejected",
                "reasons": ood_result["reasons"]
            }
        }

    # Valid in-distribution sample
    logger.info(
        f"Accepted CIFAR-100: class={class_names[predicted_idx]} ({confidence:.1f}%), "
        f"dense_sim={ood_result['max_similarity']:.3f}, backbone_sim={ood_result.get('backbone_similarity', 0.0):.3f}"
    )
    return {
        "valid": True,
        "predicted_class": str(class_names[predicted_idx]),
        "confidence": round(confidence, 2),
        "top_predictions": top_predictions,
        "top_5": top_predictions,
        "model": "EfficientNetV2B0",
        "classes": NUM_CLASSES,
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB",
        "inference_time_ms": inference_time_ms,
        "ood_metrics": {
            "max_similarity": ood_result["max_similarity"],
            "backbone_similarity": ood_result.get("backbone_similarity"),
            "nearest_prototype_class": str(class_names[ood_result["nearest_proto_idx"]]),
            "decision": "accepted"
        }
    }

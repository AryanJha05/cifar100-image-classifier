"""
OOD (Out-of-Distribution) & Invalid Image Detection Configuration
Centralizes all threshold parameters, file paths, and rejection messaging.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Path to precomputed normalized class prototype centroids
PROTOTYPES_PATH = BASE_DIR / "model" / "ood" / "cifar100_prototypes.npy"
BACKBONE_PROTOTYPES_PATH = BASE_DIR / "model" / "ood" / "cifar100_backbone_prototypes.npy"
METADATA_PATH = BASE_DIR / "model" / "ood" / "ood_metadata.json"

# =============================================================================
# Calibrated Multi-Scale Rejection Thresholds
# =============================================================================
# 1. Convolutional Backbone Prototype Similarity (avg_pool layer, 1280 dimensions):
# Captures visual, textural, and spatial domain membership.
BACKBONE_SIMILARITY_THRESHOLD = 0.25

# 2. Penultimate Dense Prototype Similarity (dense_512 layer, 512 dimensions):
# Captures fine-grained class-specific semantic alignment.
DENSE_SIMILARITY_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.45  # Alias for backward compatibility

# 3. Minimum top-1 softmax prediction confidence (percentage, 0-100).
CONFIDENCE_THRESHOLD = 35.0

# =============================================================================
# Natural Photographic vs Synthetic Graphic Domain Thresholds
# =============================================================================
# 4. Minimum per-channel color standard deviation (rejects blank/solid colors)
MIN_CHANNEL_STD = 5.0

# 5. Maximum digital vector flatness ratio (rejects UI, screenshots, schematics)
MAX_DIGITAL_FLATNESS = 0.70

# 6. Maximum digital step gradient ratio (rejects app icons, vector logos, text)
MAX_STEP_GRADIENT = 0.030

# 7. Maximum pure black/white canvas clipping ratio
MAX_EXTREME_CLIPPING = 0.65

# 8. Synthetic color gamut & vector rendering joint thresholds
SYNTHETIC_SAT_THRESHOLD = 0.38
SYNTHETIC_FLAT_THRESHOLD = 0.45

# User-facing message returned when an image fails OOD validation
REJECTION_MESSAGE = (
    "This image does not appear to belong to the CIFAR-100 classes. "
    "Please upload an image containing an object from the CIFAR-100 dataset."
)

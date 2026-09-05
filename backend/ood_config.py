"""
OOD (Out-of-Distribution) & Invalid Image Detection Configuration
Centralizes all threshold parameters, file paths, and rejection messaging.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Path to precomputed 100x512 normalized class prototype centroids
PROTOTYPES_PATH = BASE_DIR / "model" / "ood" / "cifar100_prototypes.npy"
METADATA_PATH = BASE_DIR / "model" / "ood" / "ood_metadata.json"

# =============================================================================
# Calibrated Rejection Thresholds
# =============================================================================
# Minimum cosine similarity to the nearest CIFAR-100 class prototype in 512-d space.
# Images with max cosine similarity below this value do not align with any known
# CIFAR-100 class cluster on the learned feature manifold.
SIMILARITY_THRESHOLD = 0.55

# Minimum top-1 softmax prediction confidence (expressed in percentage, 0-100).
# Even if an image has moderate cosine similarity, if the network cannot form
# a confident hypothesis, it is flagged as ambiguous/unsupported.
CONFIDENCE_THRESHOLD = 35.0

# User-facing message returned when an image fails OOD validation
REJECTION_MESSAGE = (
    "This image does not appear to belong to the CIFAR-100 classes. "
    "Please upload an image containing an object from the CIFAR-100 dataset."
)

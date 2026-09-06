"""
OOD (Out-of-Distribution) & Invalid Image Detection Configuration
Centralizes calibrated threshold parameters, file paths, and rejection messaging.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Path to precomputed normalized class prototype centroids (512-dimensional)
PROTOTYPES_PATH = BASE_DIR / "model" / "ood" / "cifar100_prototypes.npy"

# =============================================================================
# Calibrated Multi-Signal Collaborative Decision Thresholds
# =============================================================================
# High confidence regime (conf >= 55%):
# Requires genuine CIFAR-100 feature evidence:
# Either sufficient Free Energy (E >= 4.2) OR strong Prototype Similarity (S >= 0.62).
# This cleanly rejects closed-set softmax hallucinations (e.g. Document -> Worm 77.6% with E=3.53).
TIER3_MIN_CONFIDENCE = 55.0
MIN_FREE_ENERGY_HIGH_CONF = 4.2
MIN_SIMILARITY_HIGH_CONF = 0.62

# Moderate confidence regime (conf >= 35%):
# Requires balanced prototype similarity and non-trivial energy activation
TIER2_MIN_CONFIDENCE = 35.0
TIER2_MIN_SIMILARITY = 0.45
TIER2_MIN_FREE_ENERGY = 3.5

# Low confidence regime (conf >= 20%):
# Requires strong prototype alignment (e.g. difficult photographic angles or complex lighting)
TIER1_MIN_CONFIDENCE = 20.0
TIER1_MIN_SIMILARITY = 0.50
TIER1_MIN_FREE_ENERGY = 3.5

# Backward compatibility aliases
SIMILARITY_THRESHOLD = TIER2_MIN_SIMILARITY
CONFIDENCE_THRESHOLD = TIER2_MIN_CONFIDENCE

# =============================================================================
# Natural Photographic vs Synthetic / Degenerate Domain Thresholds
# =============================================================================
# Minimum per-channel color standard deviation (rejects blank/solid color images)
MIN_CHANNEL_STD = 5.0

# Minimum spatial autocorrelation between adjacent pixels (rejects random noise)
MIN_SPATIAL_CORRELATION = 0.20

# Synthetic linear gradient thresholds (rejects computer-rendered color ramps)
MAX_GRADIENT_LAPLACIAN_VAR = 1e-5
MIN_GRADIENT_FLATNESS = 0.50

# Digital document / UI canvas flatness and color concentration limits
# (Rejects text documents, spreadsheets, software dashboards, and app dialogs)
MAX_CANVAS_FLATNESS = 0.84
MAX_DARK_CANVAS_RATIO = 0.78
MAX_WHITE_CANVAS_RATIO = 0.80
MIN_CANVAS_FLATNESS = 0.55

# User-facing message returned when an image fails OOD validation
REJECTION_MESSAGE = (
    "This image does not appear to belong to the CIFAR-100 classes. "
    "Please upload an image containing an object from the CIFAR-100 dataset."
)


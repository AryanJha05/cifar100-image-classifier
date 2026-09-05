"""
OOD Detector Module
Evaluates whether an input image belongs to the CIFAR-100 data distribution
using feature-space class prototype similarity and prediction confidence.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    from backend import ood_config
except ImportError:
    import ood_config

logger = logging.getLogger("cifar100.ood")


class OODDetector:
    """
    Out-Of-Distribution and Invalid Image Detector.
    
    Architectural Rationale:
    1. Deep feature representations in the penultimate layer (`dense_512`) encode
       the semantic properties of CIFAR-100 classes after supervised transfer learning.
    2. By projecting all CIFAR-100 training samples into this 512-dimensional manifold,
       we compute the empirical normalized centroid (class prototype) for each of the
       100 target classes.
    3. When a user uploads a valid CIFAR-100 image, its normalized feature vector points
       strongly in the direction of its true class prototype, yielding a high maximum
       cosine similarity (typically > 0.60).
    4. Out-of-distribution inputs (such as text documents, UI screenshots, company logos,
       or arbitrary non-CIFAR objects) do not lie near any genuine CIFAR-100 class
       subspace, resulting in noticeably lower prototype cosine similarity.
    5. Combining prototype cosine similarity with top-1 prediction confidence produces
       a robust, real-time, non-invasive rejection boundary that protects the closed-set
       classifier from hallucinating confident answers on arbitrary user uploads.
    """

    def __init__(self, prototypes_path: Optional[str] = None):
        self.prototypes_path = prototypes_path or str(ood_config.PROTOTYPES_PATH)
        self._prototypes: Optional[np.ndarray] = None
        self._load_prototypes()

    def _load_prototypes(self) -> None:
        """Load and cache the precomputed 100x512 prototype matrix."""
        try:
            prototypes = np.load(self.prototypes_path)
            if prototypes.ndim != 2 or prototypes.shape[0] != 100:
                raise ValueError(
                    f"Expected prototypes shape (100, D), got {prototypes.shape}"
                )
            
            # Ensure all prototypes are strictly unit-length normalized
            norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self._prototypes = prototypes / norms
            logger.info(
                f"Successfully loaded {self._prototypes.shape[0]} class prototypes "
                f"(dim={self._prototypes.shape[1]}) from {self.prototypes_path}"
            )
        except Exception as e:
            logger.warning(
                f"Could not load OOD prototypes from {self.prototypes_path}: {e}. "
                "OOD feature-based rejection will fall back to confidence check."
            )
            self._prototypes = None

    @property
    def is_initialized(self) -> bool:
        return self._prototypes is not None

    def compute_similarity(self, embedding: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """
        Compute cosine similarity between the query embedding and all 100 class prototypes.

        Args:
            embedding: 1D NumPy array of shape (512,) or 2D array of shape (1, 512).

        Returns:
            max_sim: Maximum cosine similarity across all 100 classes (float in [-1, 1]).
            nearest_idx: Index of the class prototype with highest similarity (0-99).
            all_sims: 1D array of length 100 containing cosine similarities to all classes.
        """
        if self._prototypes is None:
            return 1.0, 0, np.ones(100, dtype=np.float32)

        emb = np.squeeze(embedding).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return 0.0, 0, np.zeros(self._prototypes.shape[0], dtype=np.float32)

        norm_emb = emb / norm
        similarities = np.dot(self._prototypes, norm_emb)
        nearest_idx = int(np.argmax(similarities))
        max_sim = float(similarities[nearest_idx])

        return max_sim, nearest_idx, similarities

    def evaluate(
        self,
        embedding: np.ndarray,
        confidence: float,
        sim_threshold: Optional[float] = None,
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate dual-layer OOD criteria on query embedding and prediction confidence.

        Args:
            embedding: 512-dimensional feature vector from penultimate layer `dense_512`.
            confidence: Model's top-1 softmax prediction confidence (0-100%).
            sim_threshold: Optional override for prototype similarity threshold.
            conf_threshold: Optional override for confidence threshold.

        Returns:
            dict containing:
                - is_valid (bool): True if accepted as CIFAR-100, False if rejected.
                - max_similarity (float): Cosine similarity to nearest prototype.
                - nearest_proto_idx (int): Nearest class prototype index.
                - confidence (float): Top-1 prediction confidence.
                - reasons (list[str]): Explanation reasons if rejected.
        """
        t_sim = sim_threshold if sim_threshold is not None else ood_config.SIMILARITY_THRESHOLD
        t_conf = conf_threshold if conf_threshold is not None else ood_config.CONFIDENCE_THRESHOLD

        max_sim, nearest_idx, _ = self.compute_similarity(embedding)

        reasons = []
        is_sim_valid = True
        is_conf_valid = True

        if self._prototypes is not None:
            if max_sim < t_sim:
                is_sim_valid = False
                reasons.append(
                    f"Feature similarity ({max_sim:.3f}) below threshold ({t_sim:.3f})"
                )

        if confidence < t_conf:
            is_conf_valid = False
            reasons.append(
                f"Prediction confidence ({confidence:.1f}%) below threshold ({t_conf:.1f}%)"
            )

        is_valid = is_sim_valid and is_conf_valid

        return {
            "is_valid": is_valid,
            "max_similarity": round(max_sim, 4),
            "nearest_proto_idx": nearest_idx,
            "confidence": round(confidence, 2),
            "sim_threshold": t_sim,
            "conf_threshold": t_conf,
            "reasons": reasons,
            "message": None if is_valid else ood_config.REJECTION_MESSAGE,
        }


# Singleton instance for backend reuse
_ood_detector: Optional[OODDetector] = None


def get_ood_detector() -> OODDetector:
    """Retrieve or initialize the global singleton OODDetector."""
    global _ood_detector
    if _ood_detector is None:
        _ood_detector = OODDetector()
    return _ood_detector

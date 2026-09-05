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

    def __init__(
        self,
        prototypes_path: Optional[str] = None,
        backbone_prototypes_path: Optional[str] = None
    ):
        self.prototypes_path = prototypes_path or str(ood_config.PROTOTYPES_PATH)
        self.backbone_prototypes_path = backbone_prototypes_path or str(ood_config.BACKBONE_PROTOTYPES_PATH)
        self._dense_prototypes: Optional[np.ndarray] = None
        self._backbone_prototypes: Optional[np.ndarray] = None
        self._load_prototypes()

    def _load_prototypes(self) -> None:
        """Load and cache the precomputed dense (512) and backbone (1280) prototype matrices."""
        # 1. Load dense prototypes
        try:
            prototypes = np.load(self.prototypes_path)
            if prototypes.ndim == 2 and prototypes.shape[0] == 100:
                norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self._dense_prototypes = prototypes / norms
                logger.info(
                    f"Successfully loaded {self._dense_prototypes.shape[0]} dense class prototypes "
                    f"(dim={self._dense_prototypes.shape[1]}) from {self.prototypes_path}"
                )
        except Exception as e:
            logger.warning(f"Could not load dense prototypes: {e}")
            self._dense_prototypes = None

        # 2. Load backbone prototypes
        try:
            backbone_protos = np.load(self.backbone_prototypes_path)
            if backbone_protos.ndim == 2 and backbone_protos.shape[0] == 100:
                norms = np.linalg.norm(backbone_protos, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self._backbone_prototypes = backbone_protos / norms
                logger.info(
                    f"Successfully loaded {self._backbone_prototypes.shape[0]} backbone class prototypes "
                    f"(dim={self._backbone_prototypes.shape[1]}) from {self.backbone_prototypes_path}"
                )
        except Exception as e:
            logger.warning(f"Could not load backbone prototypes: {e}")
            self._backbone_prototypes = None

    @property
    def is_initialized(self) -> bool:
        return self._dense_prototypes is not None or self._backbone_prototypes is not None

    def compute_similarity(
        self,
        embedding: np.ndarray,
        use_backbone: bool = False
    ) -> Tuple[float, int, np.ndarray]:
        """
        Compute cosine similarity between the query embedding and class prototypes.
        """
        protos = self._backbone_prototypes if use_backbone else self._dense_prototypes
        if protos is None:
            return 1.0, 0, np.ones(100, dtype=np.float32)

        emb = np.squeeze(embedding).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return 0.0, 0, np.zeros(protos.shape[0], dtype=np.float32)

        norm_emb = emb / norm
        similarities = np.dot(protos, norm_emb)
        nearest_idx = int(np.argmax(similarities))
        max_sim = float(similarities[nearest_idx])

        return max_sim, nearest_idx, similarities

    @staticmethod
    def evaluate_image_domain(pil_img: Any) -> Tuple[bool, Optional[str]]:
        """
        Evaluate natural photographic image domain criteria:
        1. Degenerate / blank content (uniform pixel values)
        2. Digital vector flatness (UI screenshot / diagram / document flat fields)
        3. Digital graphic step edges (app icons / vector logos / rendered text)
        4. Synthetic color gamut & vector fills (electric neon saturation)
        5. Synthetic mathematical linear gradient (zero photographic texture noise)
        6. Extreme background clipping (pure digital white or black canvas)
        """
        if pil_img is None:
            return True, None

        arr = np.array(pil_img.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0

        # 1. Variance / Degeneracy
        stds = np.std(arr * 255.0, axis=(0, 1))
        if np.all(stds < ood_config.MIN_CHANNEL_STD):
            return False, "Degenerate image: uniform or blank content"

        # 2. Digital Gradients
        gx = np.abs(arr[:, 1:, :] - arr[:, :-1, :])
        gy = np.abs(arr[1:, :, :] - arr[:-1, :, :])
        gx_c = gx[:-1, :, :]
        gy_c = gy[:, :-1, :]

        # 3. Digital Vector Flatness (fraction of adjacent pixels with diff <= 1.5/255)
        flat_ratio = float((np.mean(gx_c < 1.5 / 255.0) + np.mean(gy_c < 1.5 / 255.0)) / 2.0)
        if flat_ratio > ood_config.MAX_DIGITAL_FLATNESS:
            return False, f"Digital vector/UI flatness ({flat_ratio:.2f} > {ood_config.MAX_DIGITAL_FLATNESS:.2f})"

        # 4. Digital Graphic Step Edges (sharp vector contours, text glyphs, app icon outlines)
        step_grad = float(np.mean((gx_c > 0.25) | (gy_c > 0.25)))
        if step_grad > ood_config.MAX_STEP_GRADIENT:
            return False, f"Digital graphic/icon step edges ({step_grad:.3f} > {ood_config.MAX_STEP_GRADIENT:.3f})"

        # 5. Synthetic Color Gamut & Flat Vector Rendering
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        s = (mx - mn) / (mx + 1e-6)
        sat_75 = float(np.mean(s > 0.75))
        if sat_75 > ood_config.SYNTHETIC_SAT_THRESHOLD and flat_ratio > ood_config.SYNTHETIC_FLAT_THRESHOLD:
            return False, f"Synthetic graphic color and flat vector rendering (sat={sat_75:.2f}, flat={flat_ratio:.2f})"

        # 6. Synthetic Linear Color Gradient (curvature / second derivative variance)
        d2_x = np.abs(arr[:, 2:, :] - 2 * arr[:, 1:-1, :] + arr[:, :-2, :])
        d2_y = np.abs(arr[2:, :, :] - 2 * arr[1:-1, :, :] + arr[:-2, :, :])
        lap_var = float(np.var(d2_x) + np.var(d2_y))
        if lap_var < 1e-5 and flat_ratio > 0.50:
            return False, "Synthetic artificial gradient: zero photographic texture noise"

        # 7. Extreme Digital Canvas Clipping
        extreme = float(np.mean((mx < 0.04) | (mn > 0.96)))
        if extreme > ood_config.MAX_EXTREME_CLIPPING:
            return False, f"Excessive digital background clipping ({extreme:.2f} > {ood_config.MAX_EXTREME_CLIPPING:.2f})"

        return True, None

    def evaluate(
        self,
        dense_embedding: np.ndarray,
        confidence: float,
        backbone_embedding: Optional[np.ndarray] = None,
        pil_img: Optional[Any] = None,
        dense_sim_threshold: Optional[float] = None,
        backbone_sim_threshold: Optional[float] = None,
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate multi-scale OOD criteria:
        1. Natural photographic image domain check (rejects UI, screenshots, icons, text, synthetic graphics)
        2. Convolutional backbone prototype similarity (captures natural visual manifold membership)
        3. Penultimate dense prototype similarity (captures class-specific semantic alignment)
        4. Prediction confidence (guarantees categorical clarity)
        """
        t_dense = dense_sim_threshold if dense_sim_threshold is not None else ood_config.DENSE_SIMILARITY_THRESHOLD
        t_backbone = backbone_sim_threshold if backbone_sim_threshold is not None else ood_config.BACKBONE_SIMILARITY_THRESHOLD
        t_conf = conf_threshold if conf_threshold is not None else ood_config.CONFIDENCE_THRESHOLD

        reasons = []
        is_domain_valid = True
        is_dense_valid = True
        is_backbone_valid = True
        is_conf_valid = True

        # Check natural photographic image domain
        if pil_img is not None:
            domain_ok, domain_reason = self.evaluate_image_domain(pil_img)
            if not domain_ok:
                is_domain_valid = False
                reasons.append(domain_reason)

        dense_max_sim, nearest_idx, _ = self.compute_similarity(dense_embedding, use_backbone=False)

        # Check dense prototype similarity
        if self._dense_prototypes is not None:
            if dense_max_sim < t_dense:
                is_dense_valid = False
                reasons.append(
                    f"Semantic similarity ({dense_max_sim:.3f}) below threshold ({t_dense:.3f})"
                )

        # Check convolutional backbone prototype similarity
        backbone_max_sim = 1.0
        if backbone_embedding is not None and self._backbone_prototypes is not None:
            backbone_max_sim, _, _ = self.compute_similarity(backbone_embedding, use_backbone=True)
            if backbone_max_sim < t_backbone:
                is_backbone_valid = False
                reasons.append(
                    f"Visual feature similarity ({backbone_max_sim:.3f}) below threshold ({t_backbone:.3f})"
                )

        # Check softmax confidence
        if confidence < t_conf:
            is_conf_valid = False
            reasons.append(
                f"Prediction confidence ({confidence:.1f}%) below threshold ({t_conf:.1f}%)"
            )

        is_valid = is_domain_valid and is_dense_valid and is_backbone_valid and is_conf_valid

        return {
            "is_valid": is_valid,
            "max_similarity": round(dense_max_sim, 4),
            "backbone_similarity": round(backbone_max_sim, 4),
            "nearest_proto_idx": nearest_idx,
            "confidence": round(confidence, 2),
            "dense_threshold": t_dense,
            "backbone_threshold": t_backbone,
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

"""
OOD Detector Module
Evaluates whether an input image belongs to the CIFAR-100 data distribution
using feature-space class prototype similarity and collaborative prediction confidence.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from backend import ood_config
except ImportError:
    import ood_config

logger = logging.getLogger("cifar100.ood")


class OODDetector:
    """
    Out-Of-Distribution and Invalid Image Detector.

    Collaborative Decision Rule:
    1. Rejects non-photographic / degenerate inputs:
       - Blank or solid color images (channel standard deviation < 5.0)
       - Pure synthetic noise (adjacent pixel autocorrelation < 0.20)
       - Synthetic linear gradients (zero photographic texture variance)
       - Extreme document / UI canvases (dense flat monochrome background)
    2. Uses penultimate 512-dimensional class prototypes from `dense_512` to evaluate
       semantic manifold membership.
    3. Evaluates a collaborative multi-tier decision rule that prevents false rejection
       of real-world photographs while keeping closed-set hallucination protection.
    """

    def __init__(self, prototypes_path: Optional[str] = None):
        self.prototypes_path = prototypes_path or str(ood_config.PROTOTYPES_PATH)
        self._prototypes: Optional[np.ndarray] = None
        self._load_prototypes()

    def _load_prototypes(self) -> None:
        """Load and normalize precomputed 512-d class prototype centroids."""
        try:
            prototypes = np.load(self.prototypes_path)
            if prototypes.ndim == 2 and prototypes.shape[0] == 100:
                norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self._prototypes = prototypes / norms
                logger.info(
                    f"Successfully loaded {self._prototypes.shape[0]} class prototypes "
                    f"(dim={self._prototypes.shape[1]}) from {self.prototypes_path}"
                )
            else:
                logger.warning(f"Unexpected prototype matrix shape: {prototypes.shape}")
        except Exception as e:
            logger.warning(f"Could not load class prototypes from {self.prototypes_path}: {e}")
            self._prototypes = None

    @property
    def is_initialized(self) -> bool:
        return self._prototypes is not None

    def compute_similarity(self, embedding: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """Compute cosine similarities between query embedding and the 100 class prototypes."""
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

    @staticmethod
    def evaluate_image_domain(pil_img: Any) -> Tuple[bool, Optional[str]]:
        """
        Robust domain filtering:
        1. Degenerate / blank content (uniform pixel values across channels)
        2. Pure synthetic noise (near-zero spatial correlation between neighbor pixels)
        3. Synthetic mathematical linear gradients (zero curvature variance)
        4. Digital document / UI canvases (dense monochrome page with high flatness)
        """
        if pil_img is None:
            return True, None

        # Check raw image array
        arr_raw = np.array(pil_img.convert("RGB"), dtype=np.float32)

        # 1. Blank / Degenerate Check
        ch_stds = np.std(arr_raw, axis=(0, 1))
        if np.all(ch_stds < ood_config.MIN_CHANNEL_STD):
            return False, f"Degenerate / blank image (channel std {float(np.min(ch_stds)):.1f} < {ood_config.MIN_CHANNEL_STD})"

        # 2. Random Noise Check (Autocorrelation)
        gray = np.array(pil_img.convert("L"), dtype=np.float32)
        p1 = gray[:, :-1].flatten()
        p2 = gray[:, 1:].flatten()
        if np.std(p1) > 0 and np.std(p2) > 0:
            corr = float(np.corrcoef(p1, p2)[0, 1])
            if corr < ood_config.MIN_SPATIAL_CORRELATION:
                return False, f"Synthetic noise pattern (spatial correlation {corr:.3f} < {ood_config.MIN_SPATIAL_CORRELATION})"

        # 96x96 spatial inspection
        img_96 = pil_img.resize((96, 96))
        arr_96 = np.array(img_96, dtype=np.float32) / 255.0

        # 3. Mathematical Gradient Check (zero curvature variance)
        d2_x = np.abs(arr_96[:, 2:, :] - 2 * arr_96[:, 1:-1, :] + arr_96[:, :-2, :])
        d2_y = np.abs(arr_96[2:, :, :] - 2 * arr_96[1:-1, :, :] + arr_96[:-2, :, :])
        lap_var = float(np.var(d2_x) + np.var(d2_y))

        gx = np.abs(arr_96[:, 1:, :] - arr_96[:, :-1, :])
        gy = np.abs(arr_96[1:, :, :] - arr_96[:-1, :, :])
        flat_ratio = float((np.mean(gx[:-1, :, :] < 2.0 / 255.0) + np.mean(gy[:, :-1, :] < 2.0 / 255.0)) / 2.0)

        if lap_var < ood_config.MAX_GRADIENT_LAPLACIAN_VAR and flat_ratio > ood_config.MIN_GRADIENT_FLATNESS:
            return False, f"Synthetic linear gradient (zero texture variance {lap_var:.6f})"

        # 4. Digital Document / Vector / UI Canvas Check
        r, g, b = arr_96[:, :, 0], arr_96[:, :, 1], arr_96[:, :, 2]
        white_ratio = float(np.mean((r > 0.82) & (g > 0.82) & (b > 0.82)))
        dark_ratio = float(np.mean((r < 0.18) & (g < 0.18) & (b < 0.18)))

        # In natural photos, even with studio backdrops, flat_ratio is tempered by photographic texture.
        # In software dashboards, documents, spreadsheets, and UI canvases, high color concentration aligns with flatness.
        if (
            flat_ratio > ood_config.MAX_CANVAS_FLATNESS
            or (dark_ratio > ood_config.MAX_DARK_CANVAS_RATIO and flat_ratio > ood_config.MIN_CANVAS_FLATNESS)
            or (white_ratio > ood_config.MAX_WHITE_CANVAS_RATIO and flat_ratio > 0.60)
        ):
            return False, f"Document / UI canvas structure (flat={flat_ratio:.2f}, white={white_ratio:.2f}, dark={dark_ratio:.2f})"

        return True, None

    def evaluate(
        self,
        dense_embedding: np.ndarray,
        confidence: float,
        predicted_idx: Optional[int] = None,
        pil_img: Optional[Any] = None,
        logits: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Collaborative Multi-Signal Decision Rule:
        1. Checks non-photographic image domain (blank, noise, gradient, UI canvas).
        2. Computes cosine similarity of embedding against class prototypes.
        3. Computes Free Energy from raw pre-softmax logits if available:
           E = log(sum(exp(logits))).
        4. Evaluates collaborative evidence:
           - High confidence (conf >= 55%): Requires genuine feature evidence
             (Free Energy >= 4.2 OR Prototype Similarity >= 0.62)
           - Moderate confidence (conf >= 35%): Requires Sim >= 0.45 and Free Energy >= 3.5
           - Low confidence (conf >= 20%): Requires Sim >= 0.50 and Free Energy >= 3.5
        """
        reasons: List[str] = []

        # 1. Domain check
        if pil_img is not None:
            domain_ok, domain_reason = self.evaluate_image_domain(pil_img)
            if not domain_ok:
                return {
                    "is_valid": False,
                    "max_similarity": 0.0,
                    "nearest_proto_idx": predicted_idx or 0,
                    "confidence": round(confidence, 2),
                    "free_energy": 0.0,
                    "reasons": [domain_reason],
                    "message": ood_config.REJECTION_MESSAGE,
                }

        # 2. Prototype Similarity
        max_sim, nearest_idx, similarities = self.compute_similarity(dense_embedding)
        eval_sim = float(similarities[predicted_idx]) if predicted_idx is not None else max_sim

        # 3. Free Energy from Logits
        free_energy = 0.0
        if logits is not None:
            l = np.squeeze(logits).astype(np.float64)
            max_l = np.max(l)
            free_energy = float(np.log(np.sum(np.exp(l - max_l))) + max_l)

        # 4. Multi-Signal Collaborative Decision Rule
        is_valid = False
        if confidence >= ood_config.TIER3_MIN_CONFIDENCE:
            # High confidence requires genuine CIFAR-100 feature evidence:
            # Either sufficient Free Energy OR strong Prototype Similarity.
            # Prevents closed-set softmax hallucinations on out-of-domain images.
            if logits is not None and free_energy < ood_config.MIN_FREE_ENERGY_HIGH_CONF and eval_sim < ood_config.MIN_SIMILARITY_HIGH_CONF:
                reasons.append(
                    f"Low semantic feature evidence for high-confidence prediction "
                    f"(free_energy={free_energy:.2f} < {ood_config.MIN_FREE_ENERGY_HIGH_CONF}, "
                    f"similarity={eval_sim:.3f} < {ood_config.MIN_SIMILARITY_HIGH_CONF})"
                )
            else:
                is_valid = True
        elif confidence >= ood_config.TIER2_MIN_CONFIDENCE:
            if eval_sim >= ood_config.TIER2_MIN_SIMILARITY and (logits is None or free_energy >= ood_config.TIER2_MIN_FREE_ENERGY):
                is_valid = True
            else:
                reasons.append(
                    f"Insufficient semantic alignment / activation energy "
                    f"(similarity={eval_sim:.3f}, free_energy={free_energy:.2f}, confidence={confidence:.1f}%)"
                )
        elif confidence >= ood_config.TIER1_MIN_CONFIDENCE:
            if eval_sim >= ood_config.TIER1_MIN_SIMILARITY and (logits is None or free_energy >= ood_config.TIER1_MIN_FREE_ENERGY):
                is_valid = True
            else:
                reasons.append(
                    f"Low semantic prototype alignment for modest confidence "
                    f"(similarity={eval_sim:.3f}, confidence={confidence:.1f}%)"
                )
        else:
            reasons.append(f"Confidence below minimum threshold ({confidence:.1f}% < {ood_config.TIER1_MIN_CONFIDENCE}%)")

        return {
            "is_valid": is_valid,
            "max_similarity": round(eval_sim, 4),
            "nearest_proto_idx": nearest_idx,
            "confidence": round(confidence, 2),
            "free_energy": round(free_energy, 2),
            "reasons": reasons,
            "message": None if is_valid else ood_config.REJECTION_MESSAGE,
        }


# Global singleton instance
_ood_detector: Optional[OODDetector] = None


def get_ood_detector() -> OODDetector:
    """Retrieve or initialize the global singleton OODDetector."""
    global _ood_detector
    if _ood_detector is None:
        _ood_detector = OODDetector()
    return _ood_detector


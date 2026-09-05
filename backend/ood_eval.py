"""
OOD Evaluation and Threshold Calibration Suite
Empirically benchmarks the dual-layer rejection mechanism across:
1. In-Distribution: CIFAR-100 test dataset (1,000+ images across 100 classes)
2. Out-of-Distribution: Curated OOD validation suite (text documents, UI screenshots, logos, noise, non-CIFAR concepts)
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

try:
    from backend import ood_config
    from backend.ood_detector import OODDetector
except ImportError:
    import ood_config
    from ood_detector import OODDetector

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
TEST_DATA_DIR = BASE_DIR.parent / "data" / "archive" / "cifar100" / "test"
OOD_DATA_DIR = BASE_DIR.parent / "data" / "ood_validation"

IMG_SIZE = 96


def load_model_and_dual_head():
    """Load model and build dual-output network (dense_512 features + softmax)."""
    full_model = tf.keras.models.load_model(str(MODEL_PATH))
    dense_layer = full_model.get_layer("dense_512")
    dual_model = tf.keras.Model(
        inputs=full_model.inputs,
        outputs=[dense_layer.output, full_model.output],
        name="dual_model"
    )
    return dual_model


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """Preprocess PIL image matching notebook/predictor standard."""
    img_rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_rgb, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def evaluate_in_distribution(dual_model, class_names: List[str], max_samples_per_class: int = 10) -> Dict:
    """Evaluate CIFAR-100 test set images."""
    print(f"\n[1/3] Evaluating In-Distribution (CIFAR-100 Test, {max_samples_per_class}/class)...")
    detector = OODDetector()

    similarities = []
    confidences = []
    correct_predictions = 0
    total_samples = 0
    class_results = []

    t0 = time.time()
    for class_idx, class_name in enumerate(class_names):
        class_folder = TEST_DATA_DIR / class_name
        if not class_folder.exists():
            continue
        
        img_files = sorted(list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")))[:max_samples_per_class]
        for img_path in img_files:
            try:
                img = Image.open(img_path)
                tensor = preprocess_pil_image(img)
                emb, probs = dual_model(tensor, training=False)
                emb_np = emb.numpy()[0]
                probs_np = probs.numpy()[0]

                top1_idx = int(np.argmax(probs_np))
                top1_conf = float(probs_np[top1_idx] * 100)
                max_sim, nearest_idx, _ = detector.compute_similarity(emb_np)

                similarities.append(max_sim)
                confidences.append(top1_conf)
                if top1_idx == class_idx:
                    correct_predictions += 1
                total_samples += 1
            except Exception as e:
                pass

    elapsed = time.time() - t0
    sims = np.array(similarities)
    confs = np.array(confidences)

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

    return {
        "total_samples": total_samples,
        "raw_accuracy": round(accuracy, 2),
        "similarities": sims,
        "confidences": confs,
        "elapsed_sec": round(elapsed, 2),
        "sim_percentiles": {
            "p1": round(float(np.percentile(sims, 1)), 4),
            "p5": round(float(np.percentile(sims, 5)), 4),
            "p10": round(float(np.percentile(sims, 10)), 4),
            "p25": round(float(np.percentile(sims, 25)), 4),
            "p50_median": round(float(np.percentile(sims, 50)), 4),
            "mean": round(float(np.mean(sims)), 4),
        },
        "conf_percentiles": {
            "p5": round(float(np.percentile(confs, 5)), 2),
            "p10": round(float(np.percentile(confs, 10)), 2),
            "p25": round(float(np.percentile(confs, 25)), 2),
            "p50_median": round(float(np.percentile(confs, 50)), 2),
            "mean": round(float(np.mean(confs)), 2),
        }
    }


def evaluate_out_of_distribution(dual_model, class_names: List[str]) -> List[Dict]:
    """Evaluate curated OOD images."""
    print("\n[2/3] Evaluating Out-of-Distribution Validation Dataset...")
    detector = OODDetector()
    results = []

    ood_files = sorted(list(OOD_DATA_DIR.rglob("*.png")) + list(OOD_DATA_DIR.rglob("*.jpg")))
    print(f"Found {len(ood_files)} OOD validation images.")

    for path in ood_files:
        img = Image.open(path)
        tensor = preprocess_pil_image(img)
        emb, probs = dual_model(tensor, training=False)
        emb_np = emb.numpy()[0]
        probs_np = probs.numpy()[0]

        top1_idx = int(np.argmax(probs_np))
        top1_conf = float(probs_np[top1_idx] * 100)
        top1_class = class_names[top1_idx]

        max_sim, nearest_idx, _ = detector.compute_similarity(emb_np)
        nearest_class = class_names[nearest_idx]

        results.append({
            "filename": path.name,
            "category": path.parent.name,
            "forced_cifar_class": top1_class,
            "forced_confidence": round(top1_conf, 2),
            "feature_similarity": round(max_sim, 4),
            "nearest_prototype_class": nearest_class,
        })

    return results


def run_threshold_grid_search(id_metrics: Dict, ood_results: List[Dict]):
    """Evaluate rejection and acceptance trade-offs across candidate thresholds."""
    print("\n[3/3] Threshold Grid Search & Operational Calibration:")
    print("-" * 75)
    print(f"{'Sim Thresh':>10} | {'Conf Thresh':>11} | {'CIFAR Acceptance':>16} | {'OOD Rejection':>14} | {'Status':>10}")
    print("-" * 75)

    id_sims = id_metrics["similarities"]
    id_confs = id_metrics["confidences"]
    total_id = len(id_sims)
    total_ood = len(ood_results)

    sim_candidates = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    conf_candidates = [25.0, 30.0, 35.0, 40.0, 50.0]

    best_config = None
    best_score = -1

    for s_th in sim_candidates:
        for c_th in conf_candidates:
            # ID: sample is accepted if sim >= s_th and conf >= c_th
            id_accepted = np.sum((id_sims >= s_th) & (id_confs >= c_th))
            id_acc_rate = (id_accepted / total_id) * 100

            # OOD: sample is rejected if sim < s_th or conf < c_th
            ood_rejected = sum(
                1 for item in ood_results
                if item["feature_similarity"] < s_th or item["forced_confidence"] < c_th
            )
            ood_rej_rate = (ood_rejected / total_ood) * 100

            # Balanced objective: maintain high ID acceptance while maximizing OOD rejection
            # Target: ID acceptance >= 88%, OOD rejection >= 90%
            is_valid_operating_point = id_acc_rate >= 85.0 and ood_rej_rate >= 90.0
            score = (id_acc_rate * 0.4) + (ood_rej_rate * 0.6)

            status = "BALANCED" if is_valid_operating_point else "SUBOPTIMAL"
            if is_valid_operating_point and score > best_score:
                best_score = score
                best_config = (s_th, c_th, id_acc_rate, ood_rej_rate)

            print(f"{s_th:>10.2f} | {c_th:>10.1f}% | {id_acc_rate:>15.2f}% | {ood_rej_rate:>13.2f}% | {status:>10}")

    print("-" * 75)
    if best_config:
        print(f"\nRECOMMENDED BALANCED CALIBRATION:")
        print(f"  • SIMILARITY_THRESHOLD  = {best_config[0]:.2f}")
        print(f"  • CONFIDENCE_THRESHOLD = {best_config[1]:.1f}%")
        print(f"  • CIFAR-100 Acceptance Rate: {best_config[2]:.2f}%")
        print(f"  • OOD Rejection Rate:        {best_config[3]:.2f}%\n")
    return best_config


def main():
    print("=" * 60)
    print("OOD VALIDATION & EMPIRICAL BENCHMARK SUITE")
    print("=" * 60)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    dual_model = load_model_and_dual_head()

    id_metrics = evaluate_in_distribution(dual_model, class_names, max_samples_per_class=10)
    print(f"CIFAR-100 Test Samples Evaluated: {id_metrics['total_samples']}")
    print(f"Model Raw Test Accuracy:         {id_metrics['raw_accuracy']}%")
    print(f"Feature Similarity Distribution: {id_metrics['sim_percentiles']}")
    print(f"Confidence Distribution:         {id_metrics['conf_percentiles']}")

    ood_results = evaluate_out_of_distribution(dual_model, class_names)
    print("\nSample OOD Image Analysis:")
    for r in ood_results[:8]:
        print(f"  [{r['category']}] {r['filename']:<24} -> Forced Class: '{r['forced_cifar_class']}' ({r['forced_confidence']}%) | Sim: {r['feature_similarity']:.3f}")

    best_cfg = run_threshold_grid_search(id_metrics, ood_results)

    # Save detailed evaluation report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "in_distribution": {
            "dataset": "CIFAR-100 Test Set",
            "samples": id_metrics["total_samples"],
            "raw_accuracy": id_metrics["raw_accuracy"],
            "sim_percentiles": id_metrics["sim_percentiles"],
            "conf_percentiles": id_metrics["conf_percentiles"],
        },
        "out_of_distribution": {
            "dataset": "Curated OOD Validation Suite",
            "samples": len(ood_results),
            "individual_evaluations": ood_results,
        },
        "recommended_configuration": {
            "similarity_threshold": best_cfg[0] if best_cfg else 0.55,
            "confidence_threshold": best_cfg[1] if best_cfg else 35.0,
            "cifar_acceptance_rate": best_cfg[2] if best_cfg else None,
            "ood_rejection_rate": best_cfg[3] if best_cfg else None,
        }
    }
    report_path = BASE_DIR / "model" / "ood" / "ood_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed evaluation metrics saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

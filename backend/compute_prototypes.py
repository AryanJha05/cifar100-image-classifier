"""
CIFAR-100 Class Prototypes Generator
Extracts 512-dimensional feature embeddings from the penultimate `dense_512` layer
across all CIFAR-100 training images to build normalized class centroids for OOD detection.
"""

import json
import os
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
DATA_DIR = BASE_DIR.parent / "data" / "archive" / "cifar100" / "train"
OUT_DIR = BASE_DIR / "model" / "ood"
PROTOTYPES_OUT_PATH = OUT_DIR / "cifar100_prototypes.npy"
METADATA_OUT_PATH = OUT_DIR / "ood_metadata.json"

IMG_SIZE = 96
BATCH_SIZE = 32


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Memory growth notice: {e}")


def main():
    configure_gpu()
    print("=" * 60)
    print("CIFAR-100 FEATURE PROTOTYPE GENERATION")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data dir:   {DATA_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_DIR}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Total CIFAR-100 classes: {num_classes}")

    # 1. Load trained model & construct feature extractor
    print("\n[1/4] Loading trained EfficientNetV2B0 model...")
    full_model = tf.keras.models.load_model(str(MODEL_PATH))
    
    dense_layer = full_model.get_layer("dense_512")
    embedding_dim = dense_layer.output.shape[-1]
    print(f"Feature layer: {dense_layer.name}, Embedding dimension: {embedding_dim}")

    # Feature extractor
    feature_extractor = tf.keras.Model(
        inputs=full_model.inputs,
        outputs=dense_layer.output,
        name="feature_extractor"
    )

    # 2. Load dataset in class directory order
    print(f"\n[2/4] Loading images from {DATA_DIR}...")
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )
    
    # Verify class ordering matches class_names
    dataset_classes = dataset.class_names
    if dataset_classes != class_names:
        print("Notice: Reordering dataset class indices to match class_names.json...")
        class_map = {cls: idx for idx, cls in enumerate(class_names)}
        reorder_idx = [class_map[cls] for cls in dataset_classes]
    else:
        reorder_idx = None

    # 3. Accumulate feature embeddings per class
    print("\n[3/4] Extracting embeddings on GPU...")
    class_sums = np.zeros((num_classes, embedding_dim), dtype=np.float64)
    class_counts = np.zeros((num_classes,), dtype=np.int64)

    t0 = time.time()
    total_images = 0

    for batch_images, batch_labels in dataset:
        batch_labels_np = batch_labels.numpy()
        if reorder_idx is not None:
            batch_labels_np = np.array([reorder_idx[l] for l in batch_labels_np])

        # Forward pass on GPU
        embeddings = feature_extractor(batch_images, training=False).numpy()

        # L2-normalize individual embeddings before accumulation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_embeddings = embeddings / norms

        for emb, label in zip(norm_embeddings, batch_labels_np):
            class_sums[label] += emb
            class_counts[label] += 1
            total_images += 1

    elapsed = time.time() - t0
    print(f"Extracted features for {total_images} images in {elapsed:.2f}s ({total_images/elapsed:.1f} img/s)")

    # 4. Compute and normalize class centroids
    print("\n[4/4] Computing normalized class centroids (prototypes)...")
    prototypes = np.zeros((num_classes, embedding_dim), dtype=np.float32)

    for c in range(num_classes):
        if class_counts[c] == 0:
            raise ValueError(f"Class {c} ({class_names[c]}) had 0 images!")
        centroid = class_sums[c] / class_counts[c]
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            prototypes[c] = (centroid / centroid_norm).astype(np.float32)
        else:
            prototypes[c] = centroid.astype(np.float32)

    # Verify prototype norms
    proto_norms = np.linalg.norm(prototypes, axis=1)
    print(f"Prototypes shape: {prototypes.shape}")
    print(f"Prototypes norm range: [{proto_norms.min():.4f}, {proto_norms.max():.4f}]")

    # Save prototype matrix
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(PROTOTYPES_OUT_PATH), prototypes)
    print(f"Successfully saved prototype array to: {PROTOTYPES_OUT_PATH} ({PROTOTYPES_OUT_PATH.stat().st_size / 1024:.1f} KB)")

    # Save metadata
    metadata = {
        "model_architecture": "EfficientNetV2B0",
        "feature_layer": "dense_512",
        "embedding_dim": int(embedding_dim),
        "num_classes": int(num_classes),
        "total_reference_images": int(total_images),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": class_names,
        "sample_counts_per_class": class_counts.tolist(),
    }
    with open(METADATA_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Successfully saved metadata to: {METADATA_OUT_PATH}")
    print("=" * 60)
    print("PROTOTYPE GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
CIFAR-100 Training Script
Exact execution of the ML pipeline from notebook/cifar100_training.ipynb.
Trains EfficientNetV2B0 with two-stage transfer learning on CIFAR-100 (96x96 RGB).
GPU accelerated on NVIDIA GeForce RTX 3050.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers

# Configure Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "archive" / "cifar100"
TRAIN_PATH = DATA_DIR / "train"
TEST_PATH = DATA_DIR / "test"

MODEL_OUT_DIR = BASE_DIR / "backend" / "model"
MODEL_SAVE_PATH = MODEL_OUT_DIR / "best_model.keras"
CLASS_NAMES_SAVE_PATH = MODEL_OUT_DIR / "class_names.json"

IMG_SIZE = 96
BATCH_SIZE = 64
NUM_CLASSES = 100


def configure_gpu():
    """Detect and configure available GPU devices."""
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Detected GPU Devices ({len(gpus)}):")
        for idx, gpu in enumerate(gpus):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                details = tf.config.experimental.get_device_details(gpu)
                gpu_name = details.get("device_name", gpu.name)
                print(f"  [{idx}] {gpu_name} ({gpu.name}) - Memory Growth: Enabled")
            except Exception as e:
                print(f"  [{idx}] {gpu.name} - (Config notice: {e})")
    else:
        print("No GPU detected by TensorFlow. Running on CPU.")
    print("=" * 60)


def build_model():
    """Build the exact EfficientNetV2B0 model architecture from the notebook."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.4, name="dropout_0.4")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="dense_100_softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cifar100_efficientnetv2b0")
    return model, base_model


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-100 EfficientNetV2B0 Model")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-epoch GPU verification test")
    parser.add_argument("--resume", action="store_true", help="Resume fine-tuning from existing backend/model/best_model.keras")
    args = parser.parse_args()

    configure_gpu()

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(f"Dataset path not found at {DATA_DIR}. Please check the data/ folder.")

    # 1. Load Dataset
    print("\n[1/5] Loading CIFAR-100 Dataset...")
    train_data = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_data.class_names
    print(f"Classes detected: {len(class_names)}")

    # 2. Performance Prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.prefetch(AUTOTUNE)
    test_data = test_data.prefetch(AUTOTUNE)

    # Smoke Test Mode (Small batch check for GPU verification)
    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE: Running 2 batches on GPU to verify hardware acceleration <<<")
        smoke_train = train_data.take(2)
        smoke_test = test_data.take(1)

        model, base_model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(smoke_train, validation_data=smoke_test, epochs=1, verbose=1)
        print("\n>>> GPU SMOKE TEST SUCCESSFUL! Model computed forward and backward passes on GPU. <<<")
        return

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Model Architecture Setup / Resume
    if args.resume and MODEL_SAVE_PATH.exists():
        print(f"\n[2/5] Resuming from existing checkpoint: {MODEL_SAVE_PATH}...")
        model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
        # Unfreeze backbone for fine-tuning
        for layer in model.layers:
            if "efficientnet" in layer.name.lower():
                layer.trainable = True
    else:
        print("\n[2/5] Building Fresh EfficientNetV2B0 Model...")
        model, base_model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks_phase1 = [
            tf.keras.callbacks.ModelCheckpoint(
                str(MODEL_SAVE_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                mode="max",
                verbose=1
            )
        ]

        # 4. Phase 1: Feature Extraction
        print("\n[3/5] Phase 1: Training Classification Head (10 Epochs, lr=1e-3)...")
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=10,
            callbacks=callbacks_phase1
        )
        base_model.trainable = True

    # 5. Phase 2: Fine-Tuning
    print("\n[4/5] Phase 2: Fine-Tuning Entire Network (40 Epochs, lr=1e-5)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1
        )
    ]

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=40,
        callbacks=callbacks_phase2
    )

    # 6. Final Evaluation & Export
    print("\n[5/5] Evaluating Best Model Checkpoint...")
    best_model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
    loss, acc = best_model.evaluate(test_data)
    print(f"\nFinal Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {acc * 100:.2f}%\n")

    # Save class_names.json to backend/model/
    with open(CLASS_NAMES_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # Also save to notebook/ directory for reference
    with open(BASE_DIR / "notebook" / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Artifacts successfully saved:")
    print(f"  - Model: {MODEL_SAVE_PATH}")
    print(f"  - Classes: {CLASS_NAMES_SAVE_PATH}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

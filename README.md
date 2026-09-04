# CIFAR-100 Deep Learning Image Classifier

An end-to-end deep learning image classification system trained on the **CIFAR-100** dataset, featuring an **EfficientNetV2B0** convolutional neural network backbone, a high-performance **FastAPI** inference service in **Docker**, and a **Vanilla JS / CSS3** web application.

---

## Architecture Overview

```
cifar100-image-classifier/
├── backend/
│   ├── app.py                     # FastAPI REST API endpoints (/health, /predict)
│   ├── predictor.py               # Image preprocessing (96x96 RGB) & inference engine
│   ├── requirements.txt           # Container Python dependencies
│   └── model/
│       ├── best_model.keras       # Trained Keras model artifact (EfficientNetV2B0)
│       └── class_names.json       # 100 CIFAR-100 class mapping
│
├── frontend/
│   ├── index.html                 # Semantic UI markup & interactive workspace
│   ├── css/
│   │   └── style.css              # Precision Vision Design System & responsive layouts
│   ├── js/
│   │   └── app.js                 # API communication, file handling & telemetry
│   └── assets/                    # Static brand & sample assets
│
├── notebook/
│   ├── cifar100_training.ipynb    # ML training pipeline (Source of Truth)
│   └── README.md                  # Notebook documentation
│
├── data/
│   └── archive/cifar100/          # Dataset splits (50,000 train / 10,000 test)
│       ├── train/
│       ├── test/
│       └── label_names.txt
│
├── design/                        # Stitch design reference specification
├── Dockerfile                     # Backend container definition
├── docker-compose.yml             # Container orchestration
├── requirements.txt               # Root environment & Jupyter dependencies
├── .dockerignore
└── .gitignore
```

---

## Machine Learning Pipeline

- **Dataset**: CIFAR-100 (50,000 train images, 10,000 test images across 100 fine classes)
- **Input Resolution**: `96 × 96 × 3` (RGB)
- **Data Augmentation**:
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.15)`
  - `RandomZoom(0.15)`
  - `RandomTranslation(0.1, 0.1)`
  - `RandomContrast(0.1)`
- **Base Backbone**: `EfficientNetV2B0` (Pretrained on ImageNet)
- **Classification Head**:
  - `GlobalAveragePooling2D()`
  - `BatchNormalization()`
  - `Dense(512, activation="relu")`
  - `Dropout(0.4)`
  - `Dense(100, activation="softmax")`
- **Optimizer**: Adam (`lr=1e-3` initial, `lr=1e-5` fine-tuning)
- **Loss**: `sparse_categorical_crossentropy`

---

## Quick Start

### 1. Start the Docker Backend

Run from the project root:

```bash
docker compose up -d
```

- **Backend URL**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Probe**: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Start the Frontend Application

In a separate terminal window:

```bash
python3 -m http.server 8080 --directory frontend
```

Open [http://localhost:8080](http://localhost:8080) in your web browser.

### 3. Stop the Project

```bash
# Stop Docker backend
docker compose down

# Stop frontend
# Press Ctrl + C in the frontend terminal
```

---

## Local Development & Notebook Environment

To train the model or run the Jupyter Notebook on your local machine:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows PowerShell

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebook/cifar100_training.ipynb
```

---

## API Reference

### `GET /health`
Returns backend service health status.

**Response**:
```json
{
  "status": "healthy"
}
```

### `POST /predict`
Accepts a multipart image upload (`PNG`, `JPG`, `JPEG`, `WEBP`) and returns top-5 predictions.

**Response**:
```json
{
  "predicted_class": "apple",
  "confidence": 94.28,
  "top_predictions": [
    { "class_name": "apple", "confidence": 94.28 },
    { "class_name": "orange", "confidence": 2.41 },
    { "class_name": "pear", "confidence": 1.63 },
    { "class_name": "mushroom", "confidence": 0.92 },
    { "class_name": "sweet_pepper", "confidence": 0.76 }
  ]
}
```

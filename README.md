# CIFAR-100 Deep Learning Image Classifier

An end-to-end deep learning image classification system trained on the **CIFAR-100** dataset, featuring an **EfficientNetV2B0** convolutional neural network backbone, a high-performance **FastAPI** inference service in **Docker**, and a minimalist **Editorial UI** with glassmorphism panels, ambient video background, and viewport centering.

---

## Key Highlights

- **Model Architecture**: Pretrained `EfficientNetV2B0` with transfer learning and fine-tuning on CIFAR-100 (100 fine classes).
- **Inference Engine**: Asynchronous FastAPI service with on-the-fly resizing, RGB normalization, and millisecond latency telemetry.
- **Modern Web Interface**: Minimalist editorial aesthetic pairing *Instrument Serif* and *Inter*, full-bleed ambient video background, glassmorphic panels, and exact available-viewport centering.
- **Dockerized Deployment**: Fully containerized backend and frontend with Docker Compose supporting combined or standalone service execution.

---

## Architecture Overview

```
cifar100-image-classifier/
├── backend/
│   ├── app.py                     # FastAPI REST API endpoints (/health, /predict)
│   ├── predictor.py               # Preprocessing (96x96 RGB) & model inference engine
│   ├── requirements.txt           # Containerized backend Python dependencies
│   └── model/
│       ├── best_model.keras       # Trained Keras model artifact (EfficientNetV2B0)
│       └── class_names.json       # 100 CIFAR-100 class mapping
│
├── frontend/
│   ├── index.html                 # Editorial semantic markup & classification workspace
│   ├── css/
│   │   └── style.css              # Custom editorial glass design system & responsive layout
│   ├── js/
│   │   └── app.js                 # Drag-and-drop, state machine, API client & telemetry
│   └── assets/
│       └── serene-art-hero.mp4    # High-definition ambient background video
│
├── notebook/
│   ├── cifar100_training.ipynb    # ML training pipeline (Source of Truth)
│   └── README.md                  # Notebook & experiment documentation
│
├── data/
│   └── archive/cifar100/          # Dataset splits (50,000 train / 10,000 test)
│       ├── train/
│       ├── test/
│       └── label_names.txt
│
├── Dockerfile                     # Backend container definition
├── Dockerfile.frontend            # Nginx frontend container definition
├── docker-compose.yml             # Orchestration for frontend & backend
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
- **Optimizer**: Adam (`lr=1e-3` initial training, `lr=1e-5` fine-tuning)
- **Loss Function**: `sparse_categorical_crossentropy`

---

## Quick Start with Docker

### 1. Combined Stack (Backend & Frontend)
```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down
```

- **Frontend Interface**: [http://localhost:8080](http://localhost:8080)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Backend Only
```bash
# Start backend service
docker compose up -d backend

# Stop backend service
docker compose stop backend
```

- **API Base**: [http://localhost:8000](http://localhost:8000)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Frontend Only
```bash
# Start frontend service
docker compose up -d frontend

# Stop frontend service
docker compose stop frontend
```

- **Frontend Interface**: [http://localhost:8080](http://localhost:8080)

---

## Local Development & Training

To train the model or run the Jupyter Notebook on your local machine:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows PowerShell

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the training notebook
jupyter notebook notebook/cifar100_training.ipynb
```

---

## API Reference

### `GET /health`
Returns backend health status.

**Response**:
```json
{
  "status": "healthy"
}
```

### `POST /predict`
Accepts a multipart image upload (`PNG`, `JPG`, `JPEG`, `WEBP`) and returns the top-1 predicted class, confidence percentage, and top-5 ranking.

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
  ],
  "inference_time_ms": 42
}
```

---

## Frontend Design & Features

- **State-Driven Workflow**: Seamlessly transitions across `EMPTY` (drag-and-drop zone), `SELECTED` (staged preview with metadata), `LOADING` (telemetry progress), and `RESULT` (Top-1 badge + Top-5 distribution bars).
- **Available-Viewport Centering**: Automatic header height offset calculations align the workspace card with the visible viewport center on any screen size.
- **Rock-Steady Transitions**: Structured min-height constraints prevent jarring layout jumps when switching between states.
- **Responsive & Accessible**: Optimized for desktop (1440px+), laptop (1280px), tablet (768px–1024px), and mobile (375px–480px).

---

## Author

Crafted by **[Aryan Jha](https://aryanjha05.vercel.app)**

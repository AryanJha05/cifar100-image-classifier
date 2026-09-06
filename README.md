# CIFAR-100 Image Classifier

An end-to-end deep learning image classification application trained on the **CIFAR-100** dataset. The system features a fine-tuned **EfficientNetV2B0** convolutional neural network backbone, an asynchronous **FastAPI** inference engine with **Out-Of-Distribution (OOD)** invalid-image rejection, and a modern, responsive web interface.

[![GitHub Repository](https://img.shields.io/badge/GitHub-AryanJha05%2Fcifar100--image--classifier-blue?logo=github)](https://github.com/AryanJha05/cifar100-image-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Vercel](https://img.shields.io/badge/Vercel-Frontend-black?logo=vercel)](https://vercel.com)

---

## Features

- **CIFAR-100 Classification**: Predicts across all 100 fine-grained CIFAR-100 classes (animals, vehicles, flora, everyday objects).
- **EfficientNetV2B0 Backbone**: High-efficiency convolutional neural network utilizing depthwise separable convolutions and squeeze-and-excitation blocks.
- **Two-Stage Transfer Learning**: Feature extraction followed by unfreezed top-block fine-tuning on 96×96 RGB resolution.
- **FastAPI Inference Service**: Real-time asynchronous image processing with millisecond telemetry and pre-warmed model caching.
- **Multi-Signal OOD Rejection**: Rejects non-CIFAR inputs (documents, spreadsheets, screenshots, UI mockups, and unrelated objects) using logit free energy, prototype cosine similarity, and canvas domain checks.
- **Top-5 Predictions**: Returns probability distributions and rankings for the top 5 most probable classes.
- **Modern Responsive Frontend**: Fluid viewport centering, drag-and-drop ingestion, progress telemetry, and clear error guidance.
- **Docker Support**: Production-ready, lightweight Docker containerization with GPU acceleration and CPU fallback.
- **Vercel Deployment**: Seamless frontend deployment to Vercel with configurable backend rewrites and cross-origin support.

---

## Architecture

```
                 [ User Uploads Image ]
                           │
                           ▼
                 [ Frontend Interface ] (Vercel / Nginx)
                           │  POST /predict (multipart/form-data)
                           ▼
                 [ FastAPI Inference Engine ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [ Image Preprocessing ]     [ Pre-Warmed Model ]
        (96×96 RGB)              (EfficientNetV2B0)
             │                           │
             └─────────────┬─────────────┘
                           ▼
            [ Dual Feature & Logit Extraction ]
               • Penultimate features: dense_512
               • Pre-softmax logits & Free Energy
               • Softmax probabilities: 100 classes
                           │
                           ▼
          [ Multi-Signal OOD Verification ]
               • Domain Canvas & Texture Filter
               • Class Prototype Cosine Similarity
               • Logit Free Energy Activation Check
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        [ Rejected ]                [ Accepted ]
     "Image Not Recognized"      Top-1 + Top-5 Predictions
             │                           │
             └─────────────┬─────────────┘
                           ▼
               [ JSON Response to Client ]
```

---

## Model Specifications

- **Backbone Architecture**: `EfficientNetV2B0`
- **Dataset**: CIFAR-100 (100 fine classes, 60,000 images)
- **Input Resolution**: `96 × 96 × 3` RGB
- **Optimization Strategy**: Two-stage transfer learning (Feature extraction with Adam $\text{lr}=10^{-3}$ followed by fine-tuning with $\text{lr}=10^{-4}$)
- **Measured Test Accuracy**: **74.2%** top-1 accuracy on the official 10,000-image CIFAR-100 test split
- **Model Artifact**: `backend/model/best_model.keras` (~76.5 MB, self-contained Keras model format)
- **Class Mapping**: `backend/model/class_names.json` (100 deterministic class labels)
- **OOD Prototypes**: `backend/model/ood/cifar100_prototypes.npy` (Normalized 512-dimensional class centroids)

> **Important Production Note**:  
> The machine learning model is **pre-trained**. The production backend performs inference only. Model training, dataset augmentation, and notebook execution are **never** executed during production deployment.

---

## Project Structure

```
cifar100-image-classifier/
├── backend/
│   ├── app.py                     # FastAPI application endpoints (/health, /predict)
│   ├── predictor.py               # Preprocessing (96x96), logits extraction, inference
│   ├── ood_detector.py            # Multi-signal OOD decision engine (Energy + Prototypes)
│   ├── ood_config.py              # Centralized thresholds and domain constants
│   ├── requirements.txt           # Production inference dependencies
│   ├── compute_prototypes.py      # Offline prototype generator (training utility)
│   └── model/
│       ├── best_model.keras       # Trained production model artifact (76.5 MB)
│       ├── class_names.json       # 100 CIFAR-100 class names
│       └── ood/
│           ├── cifar100_prototypes.npy  # Precomputed class centroids
│           └── ood_metadata.json        # Prototype metadata
│
├── frontend/
│   ├── index.html                 # Semantic markup & classification workspace
│   ├── css/
│   │   └── main.css               # Design system, glassmorphism, responsive styles
│   ├── js/
│   │   ├── app.js                 # Drag-and-drop, state machine, API client
│   │   └── config.js              # Optional runtime API URL configuration
│   └── assets/
│       └── serene-art-hero.mp4    # Ambient hero background video
│
├── notebook/
│   ├── cifar100_training.ipynb    # Offline training pipeline & evaluation
│   └── README.md                  # Documentation for training and reproducibility
│
├── data/                          # Offline training dataset (ignored in Git & Docker)
│   └── archive/cifar100/
│       ├── train/
│       └── test/
│
├── Dockerfile                     # Production backend Docker image
├── Dockerfile.frontend            # Local development Nginx frontend image
├── docker-compose.yml             # Local multi-container development setup
├── vercel.json                    # Vercel deployment routing & API proxy rewrites
├── .dockerignore                  # Clean production build context
├── .gitignore                     # Git ignore rules (excludes datasets & temp files)
├── .env.example                   # Environment variable template
├── requirements.txt               # Local development & Jupyter notebook dependencies
├── train.py                       # Standalone training script for RTX 3050 GPU
└── README.md                      # Production project documentation
```

---

## Local Development

### Option 1: Using Docker Compose (Recommended)

Run both the frontend and backend with a single command:

```bash
# Build and launch both containers
docker compose up --build

# Open the application:
# Frontend: http://localhost:8080
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

To stop containers:
```bash
docker compose down
```

---

### Option 2: Running Without Docker

#### 1. Backend Setup

```bash
# Navigate to project directory
cd cifar100-image-classifier

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production backend dependencies
pip install -r backend/requirements.txt

# Start the FastAPI inference service
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

#### 2. Frontend Setup

Serve the static `frontend/` directory with any static file server:

```bash
# Using Python's built-in HTTP server
cd frontend
python3 -m http.server 8080
```

Open your browser at `http://localhost:8080`.

---

## Production Deployment (GitHub → Vercel)

Due to TensorFlow's size (> 500 MB) exceeding standard serverless zip limits (250 MB), the recommended, industry-standard deployment strategy is:

1. **Frontend**: Deployed to **Vercel** via GitHub integration.
2. **Backend**: Deployed as a container (using the root `Dockerfile`) on a container hosting platform (**Render**, **Railway**, **Fly.io**, **Google Cloud Run**, or **AWS App Runner**).

### Step 1: Deploy Backend Container

Deploy the repository's `Dockerfile` to your container provider:
- **Render / Railway / Fly.io / Cloud Run**: Select *Dockerfile*, set the root directory to repository root.
- The container builds with Python 3.11 slim, installs `backend/requirements.txt`, copies `backend/`, and listens on `${PORT:-8000}`.
- Note your deployed backend URL (e.g., `https://cifar100-backend.onrender.com`).

### Step 2: Deploy Frontend to Vercel

1. Push your repository to GitHub:
   ```bash
   git push origin main
   ```
2. In the [Vercel Dashboard](https://vercel.com):
   - Click **Add New** → **Project** and import your `cifar100-image-classifier` repository.
   - **Framework Preset**: Other.
   - **Root Directory**: Leave as `./` (the root `vercel.json` routes static requests to `frontend/`).
3. Set your backend URL using **either** of the following methods:
   - **Method A (Vercel Rewrites - Recommended)**:
     In `vercel.json`, replace `https://cifar100-backend.onrender.com` with your live backend URL:
     ```json
     {
       "rewrites": [
         { "source": "/predict", "destination": "https://YOUR_BACKEND_URL/predict" },
         { "source": "/health", "destination": "https://YOUR_BACKEND_URL/health" }
       ]
     }
     ```
   - **Method B (Direct URL in `frontend/js/config.js`)**:
     Uncomment and set `window.API_BASE_URL` in `frontend/js/config.js`:
     ```javascript
     window.API_BASE_URL = "https://YOUR_BACKEND_URL";
     ```
4. Click **Deploy**. Vercel will deploy the frontend to a global edge URL (e.g., `https://cifar100.vercel.app`).

---

## Environment Variables

Copy `.env.example` to `.env` to configure your environment:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for the Uvicorn HTTP server |
| `HOST` | `0.0.0.0` | Host binding for Uvicorn |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins (e.g., `https://your-app.vercel.app,http://localhost:8080`) |
| `PYTHONUNBUFFERED`| `1` | Ensures real-time container log streaming |

---

## API Reference

### Health Probe

```http
GET /health
```

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### Image Classification & Inference

```http
POST /predict
Content-Type: multipart/form-data
```

**Request Parameters**:
- `file`: Image file (`image/jpeg`, `image/png`, `image/webp`).

**Success Response — Valid In-Distribution Sample (200 OK)**:
```json
{
  "valid": true,
  "predicted_class": "clock",
  "confidence": 57.01,
  "top_predictions": [
    { "class_name": "clock", "confidence": 57.01 },
    { "class_name": "plate", "confidence": 40.00 },
    { "class_name": "bowl", "confidence": 2.71 },
    { "class_name": "worm", "confidence": 0.08 },
    { "class_name": "pear", "confidence": 0.06 }
  ],
  "top_5": [ ... ],
  "model": "EfficientNetV2B0",
  "classes": 100,
  "input_resolution": "96x96 RGB",
  "inference_time_ms": 138.2,
  "ood_metrics": {
    "max_similarity": 0.6283,
    "free_energy": 6.99,
    "nearest_prototype_class": "clock",
    "decision": "accepted"
  }
}
```

**Success Response — Rejected Out-of-Distribution Sample (200 OK)**:
```json
{
  "valid": false,
  "predicted_class": null,
  "confidence": null,
  "top_predictions": [],
  "top_5": [],
  "message": "This image does not appear to belong to the CIFAR-100 classes. Please upload an image containing an object from the CIFAR-100 dataset.",
  "model": "EfficientNetV2B0",
  "classes": 100,
  "input_resolution": "96x96 RGB",
  "inference_time_ms": 124.5,
  "ood_metrics": {
    "max_similarity": 0.5702,
    "free_energy": 3.53,
    "nearest_prototype_class": "worm",
    "decision": "rejected",
    "reasons": [
      "Low semantic feature evidence for high-confidence prediction (free_energy=3.53 < 4.2, similarity=0.570 < 0.62)"
    ]
  }
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

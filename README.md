# CIFAR-100 Image Classifier

A CNN-based image classification web application trained on the CIFAR-100 dataset.

## Architecture

- **Frontend**: HTML5, Vanilla CSS, JavaScript (`frontend/`)
- **Backend**: FastAPI Python API (`backend/`)
- **Model**: TensorFlow / Keras CNN (`backend/model/`)
- **Notebook**: Training & Experimentation (`notebook/`)
- **Containerization**: Docker & Docker Compose (`Dockerfile`, `docker-compose.yml`)

## Getting Started

### Backend (Docker)

```bash
docker compose up --build
```

Access the API at [http://localhost:8000](http://localhost:8000) and API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Frontend

The frontend consists of static files in `frontend/` and can be served with any static web server.

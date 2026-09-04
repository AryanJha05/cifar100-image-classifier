# Use Python 3.10 slim image suitable for FastAPI & TensorFlow/Keras
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies for numerical and image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application source and model directory
COPY backend/ /app/backend/

# Expose FastAPI backend port
EXPOSE 8000

# Start Uvicorn server serving backend/app.py
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

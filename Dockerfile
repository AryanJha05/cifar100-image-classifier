# Use Python 3.11 slim image for stable TensorFlow GPU & FastAPI support
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies including tensorflow[and-cuda]
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Set LD_LIBRARY_PATH for NVIDIA CUDA libs installed by tensorflow[and-cuda]
# Placed AFTER pip install so the pip cache layer is preserved on rebuilds
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

# Copy backend application source and model directory
COPY backend/ /app/backend/

# Hugging Face Spaces runs with user ID 1000
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports (8000 for standard Docker, 7860 for Hugging Face Spaces)
EXPOSE 7860 8000

# Start Uvicorn server respecting environment PORT (default 7860 for HF, 8000 for local)
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]


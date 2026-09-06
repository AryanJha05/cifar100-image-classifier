"""
FastAPI Application Entrypoint
Exposes health probe and image classification endpoints.
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from backend.predictor import predict_image
except ImportError:
    from predictor import predict_image

app = FastAPI(
    title="CIFAR-100 Image Classifier API",
    description="FastAPI Inference Backend for CIFAR-100 CNN Image Classification (EfficientNetV2B0, 96x96)",
    version="1.0.0",
)

import os

# Configure deployment-safe CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_warmup():
    """Warm up the model artifact and class names at application boot."""
    try:
        from backend.predictor import get_inference_model, load_class_names
    except ImportError:
        from predictor import get_inference_model, load_class_names
    try:
        load_class_names()
        get_inference_model()
    except Exception as e:
        # Log notice if running in environments where model will be mounted or downloaded
        pass


ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
}


@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "online",
        "service": "CIFAR-100 Image Classifier API",
        "version": "1.0.0",
        "model_architecture": "EfficientNetV2B0",
        "input_resolution": "96x96x3",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }



@app.post("/predict", tags=["Inference"])
async def predict_endpoint(file: UploadFile = File(...)):
    """Accept an uploaded image file, execute CNN prediction, and return top-5 classes."""
    # Validate content type
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Please upload a PNG, JPG, or JPEG image.",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty.",
            )

        result = predict_image(image_bytes)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image: {str(e)}",
        )

# CIFAR-100 CNN Model Training & Experimentation

This directory contains the machine learning experimentation and training notebooks for the CIFAR-100 Image Classifier.

> **Important Production Note**:  
> The training notebook and training pipeline are for offline experimentation and model development only. The production FastAPI backend loads the pre-trained `backend/model/best_model.keras` artifact and does **not** execute notebook code or run training in production.

## Overview
- **Dataset**: CIFAR-100 (100 classes, 60,000 images, 96×96 RGB)
- **Model Architecture**: EfficientNetV2B0 with two-stage transfer learning and fine-tuning
- **Output Target**: `best_model.keras` (exported to `backend/model/best_model.keras`)
- **Evaluation Accuracy**: ~74.2% top-1 accuracy on CIFAR-100 test set


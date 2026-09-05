/**
 * CIFAR-100 Image Classifier - Client Application
 * Pure Vanilla JavaScript (Zero external UI dependencies)
 * Orchestrates image ingestion, real-time backend health, and inference telemetry.
 */

// Backend API Base URL
const API_BASE_URL = window.API_BASE_URL || 'http://localhost:8000';

// Application State Machine Constants
const STATES = {
  EMPTY: 'EMPTY',
  SELECTED: 'SELECTED',
  LOADING: 'LOADING',
  RESULT: 'RESULT',
  ERROR: 'ERROR'
};

// Application State Variables
let currentState = STATES.EMPTY;
let currentFile = null;
let currentImageDataUrl = null;
let currentImageDims = { width: 96, height: 96 };

// Real sample images residing in frontend/assets/
const SAMPLE_FILES = {
  apple: {
    path: 'assets/sample_apple.png',
    filename: 'sample_apple.png'
  },
  dolphin: {
    path: 'assets/sample_dolphin.png',
    filename: 'sample_dolphin.png'
  },
  motorcycle: {
    path: 'assets/sample_motorcycle.png',
    filename: 'sample_motorcycle.png'
  }
};

/* --------------------------------------------------------------------------
   Initialization
   -------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  initDropzoneEvents();
  checkBackendHealth();
  // Poll backend health every 10 seconds
  setInterval(checkBackendHealth, 10000);
  setApplicationState(STATES.EMPTY);
});

/* --------------------------------------------------------------------------
   Backend Health Verification
   -------------------------------------------------------------------------- */
async function checkBackendHealth() {
  const dot = document.getElementById('backend-status-dot');
  const text = document.getElementById('backend-status-text');

  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      if (dot) dot.className = 'status-dot status-dot-emerald';
      if (text) text.textContent = 'Backend Online (Port 8000)';
    } else {
      if (dot) dot.className = 'status-dot status-dot-zinc';
      if (text) text.textContent = 'Backend Initializing';
    }
  } catch (err) {
    if (dot) dot.className = 'status-dot status-dot-zinc';
    if (text) text.textContent = 'Backend Offline';
  }
}

/* --------------------------------------------------------------------------
   Drag-and-Drop & File Selection Listeners
   -------------------------------------------------------------------------- */
function initDropzoneEvents() {
  const dropzone = document.getElementById('state-empty');
  const fileInput = document.getElementById('file-input-el');

  if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        processUploadedFile(files[0]);
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        processUploadedFile(e.target.files[0]);
      }
    });
  }
}

function triggerFileSelect() {
  const input = document.getElementById('file-input-el');
  if (input) {
    input.value = '';
    input.click();
  }
}

/* --------------------------------------------------------------------------
   File Processing & Viewport Staging
   -------------------------------------------------------------------------- */
function processUploadedFile(file) {
  const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    showErrorState("Unsupported file format. Please upload a PNG, JPG, or JPEG image.");
    return;
  }

  // Maximum file size limit: 15MB
  const maxBytes = 15 * 1024 * 1024;
  if (file.size > maxBytes) {
    showErrorState("Image file size exceeds 15MB. Please choose a smaller image.");
    return;
  }

  currentFile = file;

  const reader = new FileReader();
  reader.onload = function(evt) {
    currentImageDataUrl = evt.target.result;
    
    // Inspect natural image dimensions
    const tempImg = new Image();
    tempImg.onload = function() {
      currentImageDims = {
        width: tempImg.naturalWidth || 96,
        height: tempImg.naturalHeight || 96
      };
      updateStagedMetadata(file.name, file.size, currentImageDims);
      setApplicationState(STATES.SELECTED);
    };
    tempImg.src = currentImageDataUrl;
  };
  reader.readAsDataURL(file);
}

/* --------------------------------------------------------------------------
   Real Sample Image Loading (from frontend/assets/)
   -------------------------------------------------------------------------- */
async function loadSampleImage(presetKey) {
  const preset = SAMPLE_FILES[presetKey] || SAMPLE_FILES.apple;
  
  try {
    const res = await fetch(preset.path);
    if (!res.ok) {
      throw new Error(`Sample image not found at ${preset.path}`);
    }
    const blob = await res.blob();
    const file = new File([blob], preset.filename, { type: 'image/png' });
    processUploadedFile(file);
  } catch (err) {
    console.warn("Could not stage sample image:", err);
    showErrorState(`Unable to load sample image: ${err.message}`);
  }
}

/* --------------------------------------------------------------------------
   Staged Metadata Updater
   -------------------------------------------------------------------------- */
function updateStagedMetadata(filename, bytes, dims) {
  const previewImg = document.getElementById('preview-image-node');
  if (previewImg && currentImageDataUrl) {
    previewImg.src = currentImageDataUrl;
  }

  const nameEl = document.getElementById('meta-filename');
  if (nameEl) nameEl.textContent = filename || 'image.png';

  const sizeEl = document.getElementById('meta-filesize');
  if (sizeEl) {
    const kb = (bytes / 1024).toFixed(1);
    sizeEl.textContent = `${kb} KB`;
  }

  const dimsEl = document.getElementById('meta-dims');
  if (dimsEl) {
    dimsEl.textContent = `${dims.width} × ${dims.height} px`;
  }
}

/* --------------------------------------------------------------------------
   State Machine Management
   -------------------------------------------------------------------------- */
function setApplicationState(newState) {
  currentState = newState;

  // Workspace card states
  const stateEmpty = document.getElementById('state-empty');
  const stateSelected = document.getElementById('state-selected');
  const stateLoading = document.getElementById('state-loading');
  const stateResult = document.getElementById('state-result');
  const stateError = document.getElementById('state-error');

  // Dynamic Workspace header text
  const mainTitle = document.getElementById('workspace-main-title');
  const mainDesc = document.getElementById('workspace-main-desc');

  // Hide all cards first
  if (stateEmpty) stateEmpty.classList.add('hidden');
  if (stateSelected) stateSelected.classList.add('hidden');
  if (stateLoading) stateLoading.classList.add('hidden');
  if (stateResult) stateResult.classList.add('hidden');
  if (stateError) stateError.classList.add('hidden');

  switch (newState) {
    case STATES.EMPTY:
      if (stateEmpty) stateEmpty.classList.remove('hidden');
      if (mainTitle) mainTitle.textContent = 'Upload an image to begin.';
      if (mainDesc) mainDesc.textContent = 'The trained model analyzes the image and predicts one of the 100 CIFAR-100 classes.';
      break;

    case STATES.SELECTED:
      if (stateSelected) stateSelected.classList.remove('hidden');
      if (mainTitle) mainTitle.textContent = 'Selected Image';
      if (mainDesc) mainDesc.textContent = 'Image staged and ready. Click Classify Image to execute EfficientNetV2B0 inference.';
      break;

    case STATES.LOADING:
      if (stateLoading) stateLoading.classList.remove('hidden');
      if (mainTitle) mainTitle.textContent = 'Analyzing image...';
      if (mainDesc) mainDesc.textContent = 'Running EfficientNetV2B0 inference pipeline.';
      break;

    case STATES.RESULT:
      if (stateResult) stateResult.classList.remove('hidden');
      if (mainTitle) mainTitle.textContent = 'Prediction';
      if (mainDesc) mainDesc.textContent = 'Top predictions and classification probabilities generated by fine-tuned EfficientNetV2B0.';
      break;

    case STATES.ERROR:
      if (stateError) stateError.classList.remove('hidden');
      if (mainTitle) mainTitle.textContent = 'Inference Error';
      if (mainDesc) mainDesc.textContent = 'Unable to complete image classification.';
      break;
  }
}

function resetToEmptyState() {
  currentFile = null;
  currentImageDataUrl = null;
  const fileInput = document.getElementById('file-input-el');
  if (fileInput) fileInput.value = '';
  setApplicationState(STATES.EMPTY);
}

function showErrorState(message) {
  const errMsg = document.getElementById('error-message-text');
  if (errMsg) {
    errMsg.textContent = message || "Please check your connection and try again.";
  }
  setApplicationState(STATES.ERROR);
}

/* --------------------------------------------------------------------------
   Inference Execution: POST /predict to Python Backend
   -------------------------------------------------------------------------- */
async function executePrediction() {
  if (!currentFile) {
    showErrorState("No image selected for prediction. Please upload or select an image first.");
    return;
  }

  setApplicationState(STATES.LOADING);

  const formData = new FormData();
  formData.append('file', currentFile);

  const clientStartTime = performance.now();

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    });

    const clientLatency = Math.round(performance.now() - clientStartTime);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: 'Prediction failed' }));
      throw new Error(errData.detail || `Backend responded with HTTP status ${response.status}`);
    }

    const result = await response.json();
    const effectiveLatency = result.inference_time_ms !== undefined ? result.inference_time_ms : clientLatency;
    
    renderPredictionResults(result, effectiveLatency);
    setApplicationState(STATES.RESULT);

  } catch (err) {
    console.warn("Prediction execution warning:", err.message);
    showErrorState("Unable to classify image. Please check your connection and try again.");
  }
}

/* --------------------------------------------------------------------------
   Render Prediction Telemetry & Ranking
   -------------------------------------------------------------------------- */
function renderPredictionResults(result, latencyMs) {
  const formatClassName = (name) => {
    if (!name) return '';
    return String(name).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  };

  const top1Name = formatClassName(result.predicted_class);
  const top1Conf = Number(result.confidence || 0);

  // Set thumbnail image to the analyzed image
  const resultThumb = document.getElementById('result-preview-thumb');
  if (resultThumb && currentImageDataUrl) {
    resultThumb.src = currentImageDataUrl;
  }

  // Top Predicted Class Name
  const nameEl = document.getElementById('result-class-name');
  if (nameEl) nameEl.textContent = top1Name;

  // Confidence Pill Badge
  const confText = document.getElementById('result-confidence-text');
  if (confText) confText.textContent = `${top1Conf.toFixed(2)}% Confidence`;

  // Technical Summary Footer Latency
  const latencyEl = document.getElementById('footer-latency-val');
  if (latencyEl) {
    latencyEl.textContent = `${latencyMs} ms`;
  }

  // Render Top-5 Predictions List
  const top5Container = document.getElementById('top5-predictions-container');
  const predictions = result.top_predictions || result.top_5 || [];
  
  if (top5Container && predictions.length > 0) {
    top5Container.innerHTML = predictions.slice(0, 5).map((item, idx) => {
      const rankNum = idx + 1;
      const className = formatClassName(item.class_name);
      const confNum = Number(item.confidence || 0);
      const barWidth = `${Math.min(100, Math.max(2, confNum))}%`;
      const isTop = idx === 0;

      return `
        <div class="prediction-row ${isTop ? 'top-rank' : ''}">
          <div class="pred-row-header">
            <span class="pred-class-label">
              <span class="pred-rank-num">${rankNum}.</span> ${className}
            </span>
            <span class="pred-prob-val">${confNum.toFixed(2)}%</span>
          </div>
          <div class="progress-bar-track">
            <div class="progress-bar-fill ${isTop ? '' : 'muted'}" style="width: ${barWidth};"></div>
          </div>
        </div>
      `;
    }).join('');
  }
}

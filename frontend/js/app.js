/**
 * CIFAR-100 Image Classifier - Main Application Logic
 * Pure Vanilla JavaScript (No external UI dependencies)
 * Connects to FastAPI Backend (/predict, /health), manages automatic UI state transitions,
 * handles drag-and-drop file staging, and renders real model prediction telemetry.
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
let isPixelGridOn = false;

// Sample Demonstration Presets (Embedded SVG / Data URIs for Instant Testing)
const SAMPLE_PRESETS = {
  apple: {
    name: 'sample_apple.png',
    label: 'Apple',
    url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="%23FEF2F2"/><circle cx="48" cy="54" r="30" fill="%23DC2626"/><path d="M48 24 C48 16, 56 12, 56 12" stroke="%2315803D" stroke-width="4" fill="none" stroke-linecap="round"/><ellipse cx="44" cy="50" rx="6" ry="12" fill="%23EF4444"/><ellipse cx="52" cy="50" rx="6" ry="12" fill="%23B91C1C"/></svg>'
  },
  dolphin: {
    name: 'sample_dolphin.png',
    label: 'Dolphin',
    url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="%23F0F9FF"/><path d="M18 56 C30 40, 60 38, 78 48 C70 52, 55 58, 38 64 Z" fill="%230284C7"/><polygon points="46,38 54,26 56,38" fill="%230369A1"/><circle cx="68" cy="46" r="2" fill="%23FFFFFF"/></svg>'
  },
  motorcycle: {
    name: 'sample_motorcycle.png',
    label: 'Motorcycle',
    url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" fill="%23F4F4F5"/><circle cx="28" cy="62" r="14" fill="%2327272A"/><circle cx="28" cy="62" r="7" fill="%23F4F4F5"/><circle cx="68" cy="62" r="14" fill="%2327272A"/><circle cx="68" cy="62" r="7" fill="%23F4F4F5"/><polygon points="28,62 48,46 68,62 52,62" fill="%234338CA"/><line x1="48" y1="46" x2="62" y2="34" stroke="%2318181B" stroke-width="4" stroke-linecap="round"/></svg>'
  }
};

/* --------------------------------------------------------------------------
   Initialization
   -------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  initMobileNavigation();
  initDropzoneEvents();
  checkBackendHealth();
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
   Mobile Navigation Drawer
   -------------------------------------------------------------------------- */
function initMobileNavigation() {
  const toggleBtn = document.getElementById('mobile-nav-toggle-btn');
  const drawer = document.getElementById('mobile-menu-drawer');

  if (toggleBtn && drawer) {
    toggleBtn.addEventListener('click', () => {
      drawer.classList.toggle('open');
    });

    document.querySelectorAll('.mobile-nav-link').forEach(link => {
      link.addEventListener('click', () => {
        drawer.classList.remove('open');
      });
    });
  }
}

/* --------------------------------------------------------------------------
   Drag-and-Drop & File Input Event Handlers
   -------------------------------------------------------------------------- */
function initDropzoneEvents() {
  const dropzone = document.getElementById('state-empty-container');
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
   File Processing & Staging
   -------------------------------------------------------------------------- */
function processUploadedFile(file) {
  const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    showErrorState("Unsupported file format. Please upload a JPG, JPEG, or PNG image.");
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
      updateViewportMetadata(file.name, file.size, currentImageDims);
      setApplicationState(STATES.SELECTED);
    };
    tempImg.src = currentImageDataUrl;
  };
  reader.readAsDataURL(file);
}

/* --------------------------------------------------------------------------
   Sample Presets Staging
   -------------------------------------------------------------------------- */
async function loadSampleImage(presetKey) {
  const preset = SAMPLE_PRESETS[presetKey] || SAMPLE_PRESETS.apple;
  
  // Convert Data URI to File object for backend transmission
  try {
    const res = await fetch(preset.url);
    const blob = await res.blob();
    const file = new File([blob], preset.name, { type: 'image/png' });
    processUploadedFile(file);
  } catch (err) {
    console.warn("Could not stage sample image:", err);
  }
}

/* --------------------------------------------------------------------------
   Viewport Metadata Updater
   -------------------------------------------------------------------------- */
function updateViewportMetadata(filename, bytes, dims) {
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

  const badgeEl = document.getElementById('viewport-header-badge');
  if (badgeEl) {
    badgeEl.textContent = `${dims.width} × ${dims.height} px`;
  }

  const titleEl = document.getElementById('selected-image-title');
  if (titleEl) {
    titleEl.textContent = `${filename}`;
  }

  const descEl = document.getElementById('selected-image-desc');
  if (descEl) {
    const kb = (bytes / 1024).toFixed(1);
    descEl.textContent = `File: ${filename} (${kb} KB). Image staged for 96×96 RGB conversion and CNN inference.`;
  }
}

/* --------------------------------------------------------------------------
   Application State Machine
   -------------------------------------------------------------------------- */
function setApplicationState(newState) {
  currentState = newState;

  const emptyContainer = document.getElementById('state-empty-container');
  const activeContainer = document.getElementById('state-active-container');

  const panelSelected = document.getElementById('panel-selected');
  const panelLoading = document.getElementById('panel-loading');
  const panelResult = document.getElementById('panel-result');
  const panelError = document.getElementById('panel-error');

  const scanningLaser = document.getElementById('scanning-laser');
  const statusTagText = document.getElementById('viewport-status-tag-text');
  const statusTag = document.getElementById('viewport-status-tag');

  // Hide all right panels initially
  if (panelSelected) panelSelected.classList.add('hidden');
  if (panelLoading) panelLoading.classList.add('hidden');
  if (panelResult) panelResult.classList.add('hidden');
  if (panelError) panelError.classList.add('hidden');
  if (scanningLaser) scanningLaser.classList.add('hidden');

  switch (newState) {
    case STATES.EMPTY:
      if (emptyContainer) emptyContainer.classList.remove('hidden');
      if (activeContainer) activeContainer.classList.add('hidden');
      break;

    case STATES.SELECTED:
      if (emptyContainer) emptyContainer.classList.add('hidden');
      if (activeContainer) activeContainer.classList.remove('hidden');
      if (panelSelected) panelSelected.classList.remove('hidden');
      if (statusTagText) statusTagText.textContent = 'Staged';
      break;

    case STATES.LOADING:
      if (emptyContainer) emptyContainer.classList.add('hidden');
      if (activeContainer) activeContainer.classList.remove('hidden');
      if (panelLoading) panelLoading.classList.remove('hidden');
      if (scanningLaser) scanningLaser.classList.remove('hidden');
      if (statusTagText) statusTagText.textContent = 'Inferring...';
      break;

    case STATES.RESULT:
      if (emptyContainer) emptyContainer.classList.add('hidden');
      if (activeContainer) activeContainer.classList.remove('hidden');
      if (panelResult) panelResult.classList.remove('hidden');
      if (statusTagText) statusTagText.textContent = 'Classified';
      break;

    case STATES.ERROR:
      if (emptyContainer) emptyContainer.classList.add('hidden');
      if (activeContainer) activeContainer.classList.remove('hidden');
      if (panelError) panelError.classList.remove('hidden');
      if (statusTagText) statusTagText.textContent = 'Notice';
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
  if (errMsg) errMsg.textContent = message || "Please try again. Ensure the Python backend is running and reachable.";
  setApplicationState(STATES.ERROR);
}

/* --------------------------------------------------------------------------
   Inference Execution: POST /predict to Python Backend
   -------------------------------------------------------------------------- */
async function executePrediction() {
  if (!currentFile) {
    showErrorState("No image selected for prediction.");
    return;
  }

  setApplicationState(STATES.LOADING);

  const formData = new FormData();
  formData.append('file', currentFile);

  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    });

    const elapsedMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: 'Prediction failed' }));
      throw new Error(errData.detail || `Backend responded with HTTP status ${response.status}`);
    }

    const result = await response.json();
    renderPredictionResults(result, elapsedMs);
    setApplicationState(STATES.RESULT);

  } catch (err) {
    console.warn("Prediction execution warning:", err.message);
    showErrorState(err.message || "Failed to reach inference backend at http://localhost:8000.");
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

  // Top Predicted Class
  const nameEl = document.getElementById('result-class-name');
  if (nameEl) nameEl.textContent = top1Name;

  const confText = document.getElementById('result-confidence-text');
  if (confText) confText.textContent = `${top1Conf.toFixed(2)}% Confidence`;

  // Radial SVG Gauge
  const radialPct = document.getElementById('radial-pct-text');
  if (radialPct) radialPct.textContent = `${top1Conf.toFixed(1)}%`;

  const gaugeCircle = document.getElementById('gauge-circle-val');
  if (gaugeCircle) {
    gaugeCircle.setAttribute('stroke-dasharray', `${top1Conf}, 100`);
  }

  // Latency & Meta Footer
  const latencyEl = document.getElementById('footer-latency-val');
  if (latencyEl) {
    latencyEl.textContent = `Latency: ${latencyMs}ms`;
  }

  // Render Top-5 Predictions List
  const top5Container = document.getElementById('top5-predictions-container');
  if (top5Container && result.top_predictions) {
    top5Container.innerHTML = result.top_predictions.map((item, idx) => {
      const rankStr = `0${idx + 1}`;
      const className = formatClassName(item.class_name);
      const confNum = Number(item.confidence || 0);
      const barWidth = `${Math.min(100, Math.max(2, confNum))}%`;
      const isTop = idx === 0;

      return `
        <div class="prediction-row ${isTop ? 'top-rank' : ''}">
          <div class="pred-row-header">
            <span class="pred-class-label">
              <span class="pred-rank-num">${rankStr}</span> ${className}
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

/* --------------------------------------------------------------------------
   Pixel Grid Toggle Utility
   -------------------------------------------------------------------------- */
function togglePixelGrid() {
  isPixelGridOn = !isPixelGridOn;
  const grid = document.getElementById('pixel-grid-layer');
  const label = document.getElementById('grid-toggle-text');
  
  if (grid) {
    if (isPixelGridOn) {
      grid.classList.remove('hidden');
      if (label) label.textContent = 'Grid: On';
    } else {
      grid.classList.add('hidden');
      if (label) label.textContent = 'Grid: Off';
    }
  }
}

/* --------------------------------------------------------------------------
   Taxonomy Class Filter / Search
   -------------------------------------------------------------------------- */
function filterClassBadges() {
  const searchInput = document.getElementById('class-search-input');
  const query = (searchInput?.value || '').toLowerCase().trim();
  const pills = document.querySelectorAll('.class-pill');

  pills.forEach(pill => {
    const text = pill.textContent.toLowerCase();
    if (text.includes(query)) {
      pill.classList.remove('dimmed');
      if (query.length > 0) {
        pill.classList.add('highlighted');
      } else {
        pill.classList.remove('highlighted');
      }
    } else {
      if (query.length > 0) {
        pill.classList.add('dimmed');
        pill.classList.remove('highlighted');
      } else {
        pill.classList.remove('dimmed', 'highlighted');
      }
    }
  });
}

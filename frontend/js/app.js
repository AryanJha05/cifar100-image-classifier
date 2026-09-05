/**
 * CIFAR-100 Image Classifier - Client Application
 * Minimalist Editorial UI Logic
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

/* --------------------------------------------------------------------------
   Initialization
   -------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  initDropzoneEvents();
  initCenteringAndScroll();
  checkBackendHealth();
  // Poll backend health every 10 seconds
  setInterval(checkBackendHealth, 10000);
  setApplicationState(STATES.EMPTY);
});

/* --------------------------------------------------------------------------
   Header Height & Viewport Centering Synchronization
   -------------------------------------------------------------------------- */
function updateHeaderHeight() {
  const nav = document.querySelector('nav');
  if (nav) {
    const h = nav.offsetHeight;
    document.documentElement.style.setProperty('--header-height', `${h}px`);
  }
}

function initCenteringAndScroll() {
  updateHeaderHeight();
  window.addEventListener('resize', updateHeaderHeight);

  // Intercept Start Classifying buttons for exact viewport centering
  const ctaButtons = document.querySelectorAll('#hero-cta-btn, a[href="#classifier-section"]');
  ctaButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      scrollToClassifier(true);
      if (history.pushState) {
        history.pushState(null, null, '#classifier-section');
      }
    });
  });

  // Handle direct page load with hash
  if (window.location.hash === '#classifier-section') {
    setTimeout(() => scrollToClassifier(false), 50);
  }
}

/**
 * Computes the exact scroll offset to align the active classifier workspace
 * card's visual center with the center of the available visible viewport
 * (total viewport height minus fixed navigation header height).
 */
function scrollToClassifier(smooth = true) {
  const section = document.getElementById('classifier-section');
  if (!section) return;

  const nav = document.querySelector('nav');
  const headerHeight = nav ? nav.offsetHeight : 72;
  const activeCard = section.querySelector('.workspace-card:not(.hidden)') || section;
  const cardRect = activeCard.getBoundingClientRect();
  const currentScrollY = window.pageYOffset || document.documentElement.scrollTop;
  const cardAbsoluteTop = currentScrollY + cardRect.top;
  const cardHeight = cardRect.height;
  const availableHeight = window.innerHeight - headerHeight;

  let targetScrollY;
  // If card is taller than available viewport (e.g. mobile or long result card), position top cleanly below header
  if (cardHeight >= availableHeight - 32) {
    targetScrollY = cardAbsoluteTop - headerHeight - 16;
  } else {
    // Exactly center the card in the visible viewport (between bottom of header and bottom of window)
    const cardAbsoluteCenter = cardAbsoluteTop + (cardHeight / 2);
    const viewportAvailableCenter = headerHeight + (availableHeight / 2);
    targetScrollY = cardAbsoluteCenter - viewportAvailableCenter;
  }

  targetScrollY = Math.max(0, Math.round(targetScrollY));

  window.scrollTo({
    top: targetScrollY,
    behavior: smooth ? 'smooth' : 'auto'
  });
}

/**
 * Gently keeps the active card centered or cleanly positioned below header
 * during dynamic UI state transitions without abrupt jumping.
 */
function ensureCardComfortable(smooth = true) {
  const section = document.getElementById('classifier-section');
  if (!section) return;

  const currentScrollY = window.pageYOffset || document.documentElement.scrollTop;
  // Only adjust if user is already at the classifier section
  if (currentScrollY < section.offsetTop - (window.innerHeight * 0.4)) {
    return;
  }

  requestAnimationFrame(() => {
    const nav = document.querySelector('nav');
    const headerHeight = nav ? nav.offsetHeight : 72;
    const activeCard = section.querySelector('.workspace-card:not(.hidden)');
    if (!activeCard) return;

    const cardRect = activeCard.getBoundingClientRect();
    const availableHeight = window.innerHeight - headerHeight;
    const cardHeight = cardRect.height;

    if (cardHeight >= availableHeight - 32) {
      if (cardRect.top < headerHeight + 12 || cardRect.top > headerHeight + 60) {
        const targetScrollY = Math.max(0, Math.round(currentScrollY + cardRect.top - headerHeight - 16));
        window.scrollTo({ top: targetScrollY, behavior: smooth ? 'smooth' : 'auto' });
      }
    } else {
      const cardCenter = cardRect.top + (cardHeight / 2);
      const viewportCenter = headerHeight + (availableHeight / 2);
      if (Math.abs(cardCenter - viewportCenter) > 20) {
        const targetScrollY = Math.max(0, Math.round(currentScrollY + (cardCenter - viewportCenter)));
        window.scrollTo({ top: targetScrollY, behavior: smooth ? 'smooth' : 'auto' });
      }
    }
  });
}

/* --------------------------------------------------------------------------
   Backend Health Verification
   -------------------------------------------------------------------------- */
async function checkBackendHealth() {
  const dot = document.getElementById('backend-status-dot');
  const text = document.getElementById('backend-status-text');
  const dotMobile = document.getElementById('backend-status-dot-mobile');
  const textMobile = document.getElementById('backend-status-text-mobile');

  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]';
      if (text) text.textContent = 'Backend Online';
      if (dotMobile) dotMobile.className = 'w-1.5 h-1.5 rounded-full bg-emerald-500';
      if (textMobile) textMobile.textContent = 'Online';
    } else {
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-zinc-400';
      if (text) text.textContent = 'Backend Offline';
      if (dotMobile) dotMobile.className = 'w-1.5 h-1.5 rounded-full bg-zinc-400';
      if (textMobile) textMobile.textContent = 'Offline';
    }
  } catch (err) {
    if (dot) dot.className = 'w-2 h-2 rounded-full bg-zinc-400';
    if (text) text.textContent = 'Backend Offline';
    if (dotMobile) dotMobile.className = 'w-1.5 h-1.5 rounded-full bg-zinc-400';
    if (textMobile) textMobile.textContent = 'Offline';
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

  // Dynamic Workspace subhead
  const subhead = document.getElementById('workspace-subhead');

  // Hide all cards first
  if (stateEmpty) stateEmpty.classList.add('hidden');
  if (stateSelected) stateSelected.classList.add('hidden');
  if (stateLoading) stateLoading.classList.add('hidden');
  if (stateResult) stateResult.classList.add('hidden');
  if (stateError) stateError.classList.add('hidden');

  switch (newState) {
    case STATES.EMPTY:
      if (stateEmpty) stateEmpty.classList.remove('hidden');
      if (subhead) subhead.textContent = 'Drop your image into the workspace below to run fine-tuned EfficientNetV2B0 model inference.';
      break;

    case STATES.SELECTED:
      if (stateSelected) stateSelected.classList.remove('hidden');
      if (subhead) subhead.textContent = 'Image staged. Click Classify Image to execute model evaluation.';
      break;

    case STATES.LOADING:
      if (stateLoading) stateLoading.classList.remove('hidden');
      if (subhead) subhead.textContent = 'Inference in progress... Evaluating deep visual features.';
      break;

    case STATES.RESULT:
      if (stateResult) stateResult.classList.remove('hidden');
      if (subhead) subhead.textContent = 'Top predictions and classification probabilities generated by fine-tuned EfficientNetV2B0.';
      break;

    case STATES.ERROR:
      if (stateError) stateError.classList.remove('hidden');
      if (subhead) subhead.textContent = 'An issue occurred during model inference.';
      break;
  }

  // Preserve smooth viewport positioning across state changes
  ensureCardComfortable(true);
}

/**
 * Completely resets the classifier to the initial empty state.
 * Clears current image, dimensions, thumbnail, predictions, and form inputs.
 */
function resetToEmptyState() {
  currentFile = null;
  currentImageDataUrl = null;

  const fileInput = document.getElementById('file-input-el');
  if (fileInput) fileInput.value = '';

  const previewImg = document.getElementById('preview-image-node');
  if (previewImg) previewImg.src = '';

  const resultThumb = document.getElementById('result-preview-thumb');
  if (resultThumb) resultThumb.src = '';

  const nameEl = document.getElementById('result-class-name');
  if (nameEl) nameEl.textContent = '--';

  const confText = document.getElementById('result-confidence-text');
  if (confText) confText.textContent = '--% Confidence';

  const top5Container = document.getElementById('top5-predictions-container');
  if (top5Container) top5Container.innerHTML = '';

  const latencyEl = document.getElementById('footer-latency-val');
  if (latencyEl) latencyEl.textContent = '-- ms';

  setApplicationState(STATES.EMPTY);
  scrollToClassifier(true);
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
      const rankNum = String(idx + 1).padStart(2, '0');
      const className = formatClassName(item.class_name);
      const confNum = Number(item.confidence || 0);
      const barWidth = `${Math.min(100, Math.max(2, confNum))}%`;
      const isTop = idx === 0;

      return `
        <div class="p-3.5 rounded-xl bg-white/50 border border-white/60 flex flex-col gap-1.5 shadow-sm transition-all hover:bg-white/70">
          <div class="flex items-center justify-between text-sm">
            <span class="font-medium text-[#0f172a] flex items-center gap-2">
              <span class="font-mono text-xs text-[#0f172a]/40">${rankNum}</span>
              ${className}
            </span>
            <span class="font-mono text-xs font-semibold text-[#0f172a]">${confNum.toFixed(2)}%</span>
          </div>
          <div class="prog-bar-track">
            <div class="prog-bar-fill ${isTop ? '' : 'muted'}" style="width: ${barWidth};"></div>
          </div>
        </div>
      `;
    }).join('');
  }
}

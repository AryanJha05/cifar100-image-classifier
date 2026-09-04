/**
 * CIFAR-100 Image Classifier - Classifier Workbench Controller
 * Manages UI states, image uploads, previews, telemetry, and prediction pipeline.
 * Designed with a clean API adapter boundary for future FastAPI backend integration.
 */

// Preset samples for quick demonstration
const SAMPLE_PRESETS = {
  apple: {
    name: 'sample_apple.png',
    size: '142 KB',
    predictedClass: 'Apple',
    superclass: 'Fruit & Vegetables (Superclass #04)',
    confidence: 94.28,
    inferenceTime: '18ms',
    loss: '0.124',
    roi: '(4, 4, 28, 28)',
    top5: [
      { rank: '01', name: 'Apple', prob: '94.28%', width: '94.28%' },
      { rank: '02', name: 'Orange', prob: '2.41%', width: '14%' },
      { rank: '03', name: 'Pear', prob: '1.63%', width: '9%' },
      { rank: '04', name: 'Mushroom', prob: '0.92%', width: '6%' },
      { rank: '05', name: 'Sweet Pepper', prob: '0.76%', width: '4%' }
    ],
    src: "https://lh3.googleusercontent.com/aida/AEtjO1UxtM6UphbpOJFoBaSihrwsvDAF-cu3-z9plOHEtWz_Noapuxsb8wVM0Q4tv7YCw8J4zcD7pg_Av6sNH_1VZx2IDxAGgXi2JKJQANYu5pG0OFDJg_nUhW4jccTNe8MrvLy7OptqXza3ccNiOh8scLmQGOx9x5iUnCIXRl3UuAx63KTjPtfgx4J6wvgl2YjMTN9gvt70NDgLfANwibok3YEyUSuFe1SR9Lze9sr2iT41U_lJgWJ3k6gWJw"
  },
  dolphin: {
    name: 'sample_dolphin.png',
    size: '118 KB',
    predictedClass: 'Dolphin',
    superclass: 'Aquatic Mammals (Superclass #01)',
    confidence: 89.65,
    inferenceTime: '19ms',
    loss: '0.168',
    roi: '(2, 6, 30, 26)',
    top5: [
      { rank: '01', name: 'Dolphin', prob: '89.65%', width: '89.65%' },
      { rank: '02', name: 'Whale', prob: '6.12%', width: '22%' },
      { rank: '03', name: 'Shark', prob: '2.40%', width: '12%' },
      { rank: '04', name: 'Seal', prob: '1.05%', width: '6%' },
      { rank: '05', name: 'Otter', prob: '0.78%', width: '4%' }
    ],
    src: "https://images.unsplash.com/photo-1570481662006-a3a1374699e8?w=300&auto=format&fit=crop&q=80"
  },
  motorcycle: {
    name: 'sample_motorcycle.png',
    size: '156 KB',
    predictedClass: 'Motorcycle',
    superclass: 'Vehicles 1 (Superclass #18)',
    confidence: 96.12,
    inferenceTime: '17ms',
    loss: '0.089',
    roi: '(3, 5, 29, 27)',
    top5: [
      { rank: '01', name: 'Motorcycle', prob: '96.12%', width: '96.12%' },
      { rank: '02', name: 'Bicycle', prob: '2.15%', width: '12%' },
      { rank: '03', name: 'Lawn Mower', prob: '0.94%', width: '5%' },
      { rank: '04', name: 'Pickup Truck', prob: '0.45%', width: '3%' },
      { rank: '05', name: 'Bus', prob: '0.34%', width: '2%' }
    ],
    src: "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=300&auto=format&fit=crop&q=80"
  }
};

let currentActivePreset = SAMPLE_PRESETS.apple;
let currentState = 'result';
let isGridActive = false;

document.addEventListener('DOMContentLoaded', () => {
  initClassifier();
});

function initClassifier() {
  const dropzone = document.getElementById('view-empty');
  const fileInput = document.getElementById('file-input-el');

  if (dropzone && fileInput) {
    // Drag and Drop
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
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        handleImageFile(files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageFile(e.target.files[0]);
      }
    });
  }

  // Load initial sample
  setClassifierState('result');
}

/**
 * Handle user uploaded image file
 */
function handleImageFile(file) {
  if (!file.type.startsWith('image/')) {
    setClassifierState('error');
    return;
  }

  const reader = new FileReader();
  reader.onload = function(evt) {
    const previewImg = document.getElementById('preview-image-node');
    if (previewImg) previewImg.src = evt.target.result;

    const metaStrip = document.getElementById('file-meta-strip');
    if (metaStrip) {
      const sizeKb = (file.size / 1024).toFixed(1);
      metaStrip.innerHTML = `<span>${file.name}</span><span>·</span><span>${sizeKb} KB</span>`;
    }

    const headerBadge = document.getElementById('left-header-badge');
    if (headerBadge) headerBadge.textContent = '32 × 32 px (Unanalyzed)';

    setClassifierState('selected');
  };
  reader.readAsDataURL(file);
}

/**
 * Switch classifier state ('empty' | 'selected' | 'loading' | 'result' | 'error')
 */
function setClassifierState(state) {
  currentState = state;

  // Update tab buttons
  document.querySelectorAll('.state-tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeTab = document.getElementById(`btn-state-${state}`);
  if (activeTab) activeTab.classList.add('active');

  // Viewport elements
  const viewEmpty = document.getElementById('view-empty');
  const viewImage = document.getElementById('view-image');
  const viewError = document.getElementById('view-error');
  const scanningLaser = document.getElementById('scanning-laser');
  const tensorBox = document.getElementById('tensor-detection-box');
  const headerBadge = document.getElementById('left-header-badge');

  // Hide all right panels
  ['result', 'selected', 'loading', 'empty', 'error'].forEach(p => {
    const el = document.getElementById(`panel-${p}`);
    if (el) el.classList.add('hidden');
  });

  // Show active right panel
  const targetPanel = document.getElementById(`panel-${state}`);
  if (targetPanel) targetPanel.classList.remove('hidden');

  // Viewport states
  if (state === 'empty') {
    if (viewEmpty) viewEmpty.classList.remove('hidden');
    if (viewImage) viewImage.classList.add('hidden');
    if (viewError) viewError.classList.add('hidden');
    if (headerBadge) headerBadge.textContent = 'Awaiting Upload';
  } else if (state === 'error') {
    if (viewEmpty) viewEmpty.classList.add('hidden');
    if (viewImage) viewImage.classList.add('hidden');
    if (viewError) viewError.classList.remove('hidden');
    if (headerBadge) headerBadge.textContent = 'Stream Error';
  } else if (state === 'loading') {
    if (viewEmpty) viewEmpty.classList.add('hidden');
    if (viewImage) viewImage.classList.remove('hidden');
    if (viewError) viewError.classList.add('hidden');
    if (scanningLaser) scanningLaser.classList.remove('hidden');
    if (tensorBox) tensorBox.classList.add('hidden');
    if (headerBadge) headerBadge.textContent = 'Evaluating Tensor...';
  } else if (state === 'selected') {
    if (viewEmpty) viewEmpty.classList.add('hidden');
    if (viewImage) viewImage.classList.remove('hidden');
    if (viewError) viewError.classList.add('hidden');
    if (scanningLaser) scanningLaser.classList.add('hidden');
    if (tensorBox) tensorBox.classList.add('hidden');
    if (headerBadge) headerBadge.textContent = '32 × 32 px (Ready)';
  } else if (state === 'result') {
    if (viewEmpty) viewEmpty.classList.add('hidden');
    if (viewImage) viewImage.classList.remove('hidden');
    if (viewError) viewError.classList.add('hidden');
    if (scanningLaser) scanningLaser.classList.add('hidden');
    if (tensorBox) tensorBox.classList.remove('hidden');
    if (headerBadge) headerBadge.textContent = '32 × 32 px (Rescaled)';
    renderPredictionResults(currentActivePreset);
  }
}

/**
 * Render telemetry and probabilities for a prediction result
 */
function renderPredictionResults(data) {
  const classNameEl = document.getElementById('result-class-name');
  if (classNameEl) classNameEl.textContent = data.predictedClass;

  const confValEl = document.getElementById('result-confidence-val');
  if (confValEl) confValEl.textContent = `${data.confidence}% Confidence`;

  const radialPctEl = document.getElementById('radial-pct-text');
  if (radialPctEl) radialPctEl.textContent = `${data.confidence.toFixed(1)}%`;

  const gaugeCircle = document.getElementById('gauge-circle-val');
  if (gaugeCircle) {
    gaugeCircle.setAttribute('stroke-dasharray', `${data.confidence}, 100`);
  }

  const tensorTag = document.getElementById('tensor-class-tag');
  if (tensorTag) {
    tensorTag.textContent = `cifar_class: ${data.predictedClass.toLowerCase()} [${(data.confidence/100).toFixed(4)}]`;
  }

  const tensorRoi = document.getElementById('tensor-roi-tag');
  if (tensorRoi) tensorRoi.textContent = `ROI: ${data.roi}`;

  const metaFooter = document.getElementById('result-meta-footer');
  if (metaFooter) {
    metaFooter.innerHTML = `
      <span>Superclass: ${data.superclass}</span>
      <span>Loss: ${data.loss}</span>
      <span>Latency: ${data.inferenceTime}</span>
    `;
  }

  const top5Container = document.getElementById('top5-predictions-container');
  if (top5Container && data.top5) {
    top5Container.innerHTML = data.top5.map((item, idx) => `
      <div class="prediction-row ${idx === 0 ? 'top-rank' : ''}">
        <div class="pred-row-header">
          <span class="pred-class-label">
            <span class="pred-rank-num">${item.rank}</span> ${item.name}
          </span>
          <span class="pred-prob-val">${item.prob}</span>
        </div>
        <div class="progress-bar-track">
          <div class="progress-bar-fill ${idx === 0 ? 'indigo' : 'muted'}" style="width: ${item.width};"></div>
        </div>
      </div>
    `).join('');
  }
}

/**
 * Trigger file selection dialog
 */
function triggerFileSelect() {
  const input = document.getElementById('file-input-el');
  if (input) input.click();
}

/**
 * Toggle 32x32 pixel grid overlay
 */
function togglePixelGrid() {
  isGridActive = !isGridActive;
  const gridLayer = document.getElementById('pixel-grid-layer');
  const toggleLabel = document.getElementById('grid-toggle-text');
  if (gridLayer) {
    if (isGridActive) {
      gridLayer.classList.remove('hidden');
      if (toggleLabel) toggleLabel.textContent = 'Grid: On';
    } else {
      gridLayer.classList.add('hidden');
      if (toggleLabel) toggleLabel.textContent = 'Grid: Off';
    }
  }
}

/**
 * Load a preset sample (e.g. apple, dolphin, motorcycle)
 */
function loadSamplePreset(presetKey) {
  const sample = SAMPLE_PRESETS[presetKey] || SAMPLE_PRESETS.apple;
  currentActivePreset = sample;

  const preview = document.getElementById('preview-image-node');
  if (preview) preview.src = sample.src;

  const meta = document.getElementById('file-meta-strip');
  if (meta) {
    meta.innerHTML = `<span>${sample.name}</span><span>·</span><span>${sample.size}</span>`;
  }

  setClassifierState('result');
}

/**
 * Reset back to default sample apple
 */
function resetSampleImage() {
  loadSamplePreset('apple');
}

/**
 * Simulate the CNN feedforward execution with step-by-step telemetry
 * (Ready to be swapped with real FastAPI backend call)
 */
function runInference() {
  setClassifierState('loading');
  
  // Future implementation:
  // callBackendInference(imageBlob).then(result => renderPredictionResults(result))
  
  setTimeout(() => {
    setClassifierState('result');
  }, 1200);
}

/**
 * Clean API boundary function for future FastAPI POST /predict integration
 */
async function callBackendInference(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Inference failed');
    return await response.json();
  } catch (err) {
    console.error('Inference error:', err);
    throw err;
  }
}

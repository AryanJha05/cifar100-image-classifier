/**
 * CIFAR-100 Image Classifier - Main Application Logic
 * Implements interactive workbench states, file upload/preview,
 * telemetry visualizations, sample presets, and taxonomy search.
 */

// Sample image presets for interactive demonstration
const SAMPLE_PRESETS = {
  apple: {
    name: 'input_apple_01.png',
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
    name: 'input_dolphin_01.png',
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
    name: 'input_motorcycle_01.png',
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

let currentPreset = SAMPLE_PRESETS.apple;
let currentState = 'result';
let isGridActive = false;

document.addEventListener('DOMContentLoaded', () => {
  initMobileMenu();
  initClassifierEvents();
  renderPredictionData(currentPreset);
});

/* --------------------------------------------------------------------------
   Mobile Navigation Toggle
   -------------------------------------------------------------------------- */
function initMobileMenu() {
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
   Classifier Viewport & State Machine
   -------------------------------------------------------------------------- */
function initClassifierEvents() {
  const dropzone = document.getElementById('view-empty');
  const fileInput = document.getElementById('file-input-el');

  if (dropzone && fileInput) {
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
      if (files.length > 0) {
        handleUploadedFile(files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleUploadedFile(e.target.files[0]);
      }
    });
  }
}

function handleUploadedFile(file) {
  if (!file.type.startsWith('image/')) {
    setClassifierState('error');
    return;
  }

  const reader = new FileReader();
  reader.onload = function(evt) {
    const preview = document.getElementById('preview-image-node');
    if (preview) preview.src = evt.target.result;

    const meta = document.getElementById('file-meta-strip');
    if (meta) {
      const kb = (file.size / 1024).toFixed(1);
      meta.innerHTML = `<span>${file.name}</span><span>·</span><span>${kb} KB</span>`;
    }

    const badge = document.getElementById('left-header-badge');
    if (badge) badge.textContent = '32 × 32 px (Unanalyzed)';

    setClassifierState('selected');
  };
  reader.readAsDataURL(file);
}

function setClassifierState(state) {
  currentState = state;

  // Update tabs
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

  // Panels
  ['result', 'selected', 'loading', 'empty', 'error'].forEach(p => {
    const el = document.getElementById(`panel-${p}`);
    if (el) el.classList.add('hidden');
  });

  const targetPanel = document.getElementById(`panel-${state}`);
  if (targetPanel) targetPanel.classList.remove('hidden');

  if (state === 'empty') {
    viewEmpty.classList.remove('hidden');
    viewImage.classList.add('hidden');
    viewError.classList.add('hidden');
    headerBadge.textContent = 'Awaiting Upload';
  } else if (state === 'error') {
    viewEmpty.classList.add('hidden');
    viewImage.classList.add('hidden');
    viewError.classList.remove('hidden');
    headerBadge.textContent = 'Stream Error';
  } else if (state === 'loading') {
    viewEmpty.classList.add('hidden');
    viewImage.classList.remove('hidden');
    viewError.classList.add('hidden');
    scanningLaser.classList.remove('hidden');
    tensorBox.classList.add('hidden');
    headerBadge.textContent = 'Evaluating Tensor...';
  } else if (state === 'selected') {
    viewEmpty.classList.add('hidden');
    viewImage.classList.remove('hidden');
    viewError.classList.add('hidden');
    scanningLaser.classList.add('hidden');
    tensorBox.classList.add('hidden');
    headerBadge.textContent = '32 × 32 px (Ready)';
  } else if (state === 'result') {
    viewEmpty.classList.add('hidden');
    viewImage.classList.remove('hidden');
    viewError.classList.add('hidden');
    scanningLaser.classList.add('hidden');
    tensorBox.classList.remove('hidden');
    headerBadge.textContent = '32 × 32 px (Rescaled)';
    renderPredictionData(currentPreset);
  }
}

function renderPredictionData(data) {
  const nameEl = document.getElementById('result-class-name');
  if (nameEl) nameEl.textContent = data.predictedClass;

  const confEl = document.getElementById('result-confidence-val');
  if (confEl) confEl.textContent = `${data.confidence}% Confidence`;

  const radialPct = document.getElementById('radial-pct-text');
  if (radialPct) radialPct.textContent = `${data.confidence.toFixed(1)}%`;

  const gauge = document.getElementById('gauge-circle-val');
  if (gauge) gauge.setAttribute('stroke-dasharray', `${data.confidence}, 100`);

  const tag = document.getElementById('tensor-class-tag');
  if (tag) tag.textContent = `cifar_class: ${data.predictedClass.toLowerCase()} [${(data.confidence/100).toFixed(4)}]`;

  const roi = document.getElementById('tensor-roi-tag');
  if (roi) roi.textContent = `ROI: ${data.roi}`;

  const metaFooter = document.getElementById('result-meta-footer');
  if (metaFooter) {
    metaFooter.innerHTML = `
      <span>Superclass: ${data.superclass}</span>
      <span>Loss: ${data.loss}</span>
      <span>Arch: ConvNet-v2.1</span>
    `;
  }

  const top5Box = document.getElementById('top5-predictions-container');
  if (top5Box && data.top5) {
    top5Box.innerHTML = data.top5.map((item, idx) => `
      <div class="prediction-row ${idx === 0 ? 'top-rank' : ''}">
        <div class="pred-row-header">
          <span class="pred-class-label">
            <span class="pred-rank-num">${item.rank}</span> ${item.name}
          </span>
          <span class="pred-prob-val">${item.prob}</span>
        </div>
        <div class="progress-bar-track">
          <div class="progress-bar-fill ${idx === 0 ? '' : 'muted'}" style="width: ${item.width};"></div>
        </div>
      </div>
    `).join('');
  }
}

function triggerFileSelect() {
  const input = document.getElementById('file-input-el');
  if (input) input.click();
}

function togglePixelGrid() {
  isGridActive = !isGridActive;
  const grid = document.getElementById('pixel-grid-layer');
  const label = document.getElementById('grid-toggle-text');
  if (grid) {
    if (isGridActive) {
      grid.classList.remove('hidden');
      if (label) label.textContent = 'Grid: On';
    } else {
      grid.classList.add('hidden');
      if (label) label.textContent = 'Grid: Off';
    }
  }
}

function loadSamplePreset(presetKey) {
  const sample = SAMPLE_PRESETS[presetKey] || SAMPLE_PRESETS.apple;
  currentPreset = sample;

  const preview = document.getElementById('preview-image-node');
  if (preview) preview.src = sample.src;

  const meta = document.getElementById('file-meta-strip');
  if (meta) {
    meta.innerHTML = `<span>${sample.name}</span><span>·</span><span>${sample.size}</span>`;
  }

  setClassifierState('result');
}

function resetSampleImage() {
  loadSamplePreset('apple');
}

function simulateInference() {
  setClassifierState('loading');
  setTimeout(() => {
    setClassifierState('result');
  }, 1200);
}

/* --------------------------------------------------------------------------
   Taxonomy Search Filter
   -------------------------------------------------------------------------- */
function filterClassBadges() {
  const query = (document.getElementById('class-search-input')?.value || '').toLowerCase().trim();
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

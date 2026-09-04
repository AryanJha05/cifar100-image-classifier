/**
 * CIFAR-100 Image Classifier - Architecture Page Controller
 * Handles interactive tensor flow node inspection and layer parameter highlights.
 */

document.addEventListener('DOMContentLoaded', () => {
  initArchitectureExplorer();
});

function initArchitectureExplorer() {
  const layerNodes = document.querySelectorAll('.tensor-step-node');
  const detailBox = document.getElementById('tensor-layer-detail-box');

  const LAYER_SPECS = {
    '01': {
      title: 'Input Tensor (RGB Normalized)',
      shape: '(Batch, 32, 32, 3)',
      parameters: 0,
      description: 'Raw image rescaled to 32×32 spatial resolution with 3 color channels (Red, Green, Blue). Values scaled to float32 range [0.0, 1.0].'
    },
    '02': {
      title: 'Conv2D Layer 1 + ReLU',
      shape: '(Batch, 30, 30, 16)',
      parameters: 448,
      description: '16 filters with 3×3 receptive fields extracting basic low-level spatial features (edges, contours, color gradients).'
    },
    '03': {
      title: 'Conv2D Layer 2 + MaxPool2D',
      shape: '(Batch, 14, 14, 32)',
      parameters: 4640,
      description: '32 filters (3×3) followed by 2×2 max pooling downsampling, halving spatial dimensions while preserving translation invariance.'
    },
    '04': {
      title: 'Conv2D Layer 3 + MaxPool2D',
      shape: '(Batch, 6, 6, 64)',
      parameters: 18496,
      description: '64 high-level filters detecting complex textures, object parts, and domain patterns.'
    },
    '05': {
      title: 'Dense Classification (100-way Softmax)',
      shape: '(Batch, 100)',
      parameters: 230500,
      description: 'Fully-connected dense layer with softmax activation outputting a probability distribution across all 100 CIFAR-100 fine classes.'
    }
  };

  layerNodes.forEach(node => {
    node.addEventListener('click', () => {
      layerNodes.forEach(n => n.classList.remove('active'));
      node.classList.add('active');

      const stepId = node.dataset.step;
      const spec = LAYER_SPECS[stepId];
      if (spec && detailBox) {
        detailBox.innerHTML = `
          <div class="card-header">
            <span class="card-header-title">${spec.title}</span>
            <span class="badge badge-primary font-mono">${spec.shape}</span>
          </div>
          <p class="research-text" style="margin-bottom: 8px;">${spec.description}</p>
          <div class="card-footer" style="padding-top: 8px; margin-top: 8px;">
            <span>Trainable Parameters: ${spec.parameters.toLocaleString()}</span>
            <span>Activation: ReLU / Softmax</span>
          </div>
        `;
      }
    });
  });
}

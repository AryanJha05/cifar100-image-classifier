/**
 * CIFAR-100 Image Classifier - Client Runtime Configuration
 * 
 * In local development (localhost / 127.0.0.1), app.js automatically connects to http://localhost:8000.
 * In production deployments on Vercel:
 *   - Option 1 (Default): Leave window.API_BASE_URL undefined. Requests to /predict and /health
 *     are relative paths handled by Vercel rewrites (vercel.json).
 *   - Option 2 (Direct Backend): Uncomment and set window.API_BASE_URL to your deployed backend URL:
 *     window.API_BASE_URL = "https://your-cifar100-backend.onrender.com";
 */

// Production Railway backend URL
window.API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : 'https://cifar100-image-classifier-production.up.railway.app';


/**
 * CIFAR-100 Image Classifier - Shared Navigation & Mobile Menu Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
});

function initNavigation() {
  const toggleBtn = document.getElementById('mobile-nav-toggle-btn');
  const menuDrawer = document.getElementById('mobile-menu-drawer');

  if (toggleBtn && menuDrawer) {
    toggleBtn.addEventListener('click', () => {
      const isOpen = menuDrawer.classList.toggle('open');
      toggleBtn.setAttribute('aria-expanded', isOpen);
    });

    // Close mobile menu on outside click or resize
    document.addEventListener('click', (e) => {
      if (!toggleBtn.contains(e.target) && !menuDrawer.contains(e.target)) {
        menuDrawer.classList.remove('open');
        toggleBtn.setAttribute('aria-expanded', 'false');
      }
    });

    window.addEventListener('resize', () => {
      if (window.innerWidth > 768) {
        menuDrawer.classList.remove('open');
        toggleBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Highlight active link based on current page path or body dataset
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-link');
  
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (!href) return;
    
    // Check match with relative or absolute paths
    const isCurrent = currentPath.endsWith(href) || 
                      (href.includes('classifier.html') && (currentPath.endsWith('/') || currentPath.endsWith('index.html'))) ||
                      (href.endsWith(currentPath.split('/').pop()) && currentPath.split('/').pop() !== '');

    if (isCurrent) {
      link.classList.add('active');
    }
  });
}

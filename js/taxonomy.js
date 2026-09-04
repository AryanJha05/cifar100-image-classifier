/**
 * CIFAR-100 Image Classifier - Taxonomy Explorer Controller
 * Manages the complete CIFAR-100 hierarchical dataset (20 superclasses, 100 fine classes)
 * with instant search, category filtering, and highlighted query matches.
 */

const CIFAR100_TAXONOMY = [
  {
    id: '01',
    superclass: 'Aquatic Mammals',
    category: 'aquatic',
    classes: ['Beaver', 'Dolphin', 'Otter', 'Seal', 'Whale']
  },
  {
    id: '02',
    superclass: 'Fish',
    category: 'aquatic',
    classes: ['Aquarium Fish', 'Flatfish', 'Ray', 'Shark', 'Trout']
  },
  {
    id: '03',
    superclass: 'Flowers',
    category: 'flora',
    classes: ['Orchid', 'Poppy', 'Rose', 'Sunflower', 'Tulip']
  },
  {
    id: '04',
    superclass: 'Food Containers',
    category: 'objects',
    classes: ['Bottle', 'Bowl', 'Can', 'Cup', 'Plate']
  },
  {
    id: '05',
    superclass: 'Fruit & Vegetables',
    category: 'flora',
    classes: ['Apple', 'Mushroom', 'Orange', 'Pear', 'Sweet Pepper']
  },
  {
    id: '06',
    superclass: 'Household Electrical Devices',
    category: 'objects',
    classes: ['Clock', 'Computer Keyboard', 'Lamp', 'Telephone', 'Television']
  },
  {
    id: '07',
    superclass: 'Household Furniture',
    category: 'objects',
    classes: ['Bed', 'Chair', 'Couch', 'Table', 'Wardrobe']
  },
  {
    id: '08',
    superclass: 'Insects',
    category: 'animals',
    classes: ['Bee', 'Beetle', 'Butterfly', 'Caterpillar', 'Cockroach']
  },
  {
    id: '09',
    superclass: 'Large Carnivores',
    category: 'animals',
    classes: ['Bear', 'Leopard', 'Lion', 'Tiger', 'Wolf']
  },
  {
    id: '10',
    superclass: 'Large Man-Made Outdoor Things',
    category: 'scenes',
    classes: ['Bridge', 'Castle', 'House', 'Road', 'Skyscraper']
  },
  {
    id: '11',
    superclass: 'Large Natural Outdoor Scenes',
    category: 'scenes',
    classes: ['Cloud', 'Forest', 'Mountain', 'Plain', 'Sea']
  },
  {
    id: '12',
    superclass: 'Large Omnivores & Herbivores',
    category: 'animals',
    classes: ['Camel', 'Cattle', 'Chimpanzee', 'Elephant', 'Kangaroo']
  },
  {
    id: '13',
    superclass: 'Medium-Sized Mammals',
    category: 'animals',
    classes: ['Fox', 'Porcupine', 'Possum', 'Raccoon', 'Skunk']
  },
  {
    id: '14',
    superclass: 'Non-Insect Invertebrates',
    category: 'animals',
    classes: ['Crab', 'Lobster', 'Snail', 'Spider', 'Worm']
  },
  {
    id: '15',
    superclass: 'People',
    category: 'people',
    classes: ['Baby', 'Boy', 'Girl', 'Man', 'Woman']
  },
  {
    id: '16',
    superclass: 'Reptiles',
    category: 'animals',
    classes: ['Crocodile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle']
  },
  {
    id: '17',
    superclass: 'Small Mammals',
    category: 'animals',
    classes: ['Hamster', 'Mouse', 'Rabbit', 'Shrew', 'Squirrel']
  },
  {
    id: '18',
    superclass: 'Trees',
    category: 'flora',
    classes: ['Maple Tree', 'Oak Tree', 'Palm Tree', 'Pine Tree', 'Willow Tree']
  },
  {
    id: '19',
    superclass: 'Vehicles 1 (Wheeled)',
    category: 'vehicles',
    classes: ['Bicycle', 'Bus', 'Motorcycle', 'Pickup Truck', 'Train']
  },
  {
    id: '20',
    superclass: 'Vehicles 2 (Specialized)',
    category: 'vehicles',
    classes: ['Lawn Mower', 'Rocket', 'Streetcar', 'Tank', 'Tractor']
  }
];

let activeCategoryFilter = 'all';

document.addEventListener('DOMContentLoaded', () => {
  renderTaxonomy();
  initTaxonomyFilters();
});

function initTaxonomyFilters() {
  const searchInput = document.getElementById('class-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      filterTaxonomy();
    });
  }

  const filterButtons = document.querySelectorAll('.filter-tab-btn');
  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      filterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeCategoryFilter = btn.dataset.category || 'all';
      filterTaxonomy();
    });
  });
}

function renderTaxonomy() {
  const container = document.getElementById('taxonomy-grid-container');
  if (!container) return;

  container.innerHTML = CIFAR100_TAXONOMY.map(group => `
    <div class="superclass-card" data-category="${group.category}" data-superclass="${group.superclass.toLowerCase()}">
      <div>
        <div class="superclass-header">
          <span class="superclass-title">${group.superclass}</span>
          <span class="superclass-count">${group.classes.length}/5</span>
        </div>
        <div class="class-pills-wrap">
          ${group.classes.map(cls => `
            <span class="class-pill" data-name="${cls.toLowerCase()}">${cls}</span>
          `).join('')}
        </div>
      </div>
      <div class="card-footer" style="padding-top: 8px; margin-top: 12px; font-size: 10px;">
        <span>ID: #${group.id}</span>
        <span>${group.category.toUpperCase()}</span>
      </div>
    </div>
  `).join('');
}

function filterTaxonomy() {
  const query = (document.getElementById('class-search-input')?.value || '').trim().toLowerCase();
  const cards = document.querySelectorAll('.superclass-card');
  const countStatsEl = document.getElementById('taxonomy-stats-count');

  let totalVisibleClasses = 0;
  let totalVisibleCards = 0;

  cards.forEach(card => {
    const cardCategory = card.dataset.category;
    const categoryMatches = activeCategoryFilter === 'all' || cardCategory === activeCategoryFilter;

    let hasMatchingClassInCard = false;
    const pills = card.querySelectorAll('.class-pill');

    pills.forEach(pill => {
      const className = pill.dataset.name;
      const queryMatches = query === '' || className.includes(query);

      if (queryMatches && categoryMatches) {
        pill.classList.remove('dimmed');
        if (query.length > 0) {
          pill.classList.add('highlighted');
        } else {
          pill.classList.remove('highlighted');
        }
        hasMatchingClassInCard = true;
        totalVisibleClasses++;
      } else {
        if (query.length > 0 && !queryMatches) {
          pill.classList.add('dimmed');
          pill.classList.remove('highlighted');
        } else {
          pill.classList.remove('dimmed', 'highlighted');
        }
      }
    });

    if (categoryMatches && (query === '' || hasMatchingClassInCard)) {
      card.classList.remove('hidden');
      totalVisibleCards++;
    } else {
      card.classList.add('hidden');
    }
  });

  if (countStatsEl) {
    countStatsEl.textContent = `Showing ${totalVisibleClasses} classes across ${totalVisibleCards} superclasses`;
  }
}

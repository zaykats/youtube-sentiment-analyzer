// popup.js - Logique principale de l'extension

// Configuration de l'API
const API_URL = 'http://localhost:8000'; // Changer pour l'URL de production

// État de l'application
let currentFilter = 'all';
let allPredictions = [];

// Éléments DOM
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const statsSection = document.getElementById('statsSection');
const filtersSection = document.getElementById('filtersSection');
const commentsSection = document.getElementById('commentsSection');
const errorSection = document.getElementById('errorSection');
const themeToggle = document.getElementById('themeToggle');
const copyBtn = document.getElementById('copyBtn');

// Initialisation
document.addEventListener('DOMContentLoaded', () => {
  loadTheme();
  setupEventListeners();
});

// Configuration des écouteurs d'événements
function setupEventListeners() {
  analyzeBtn.addEventListener('click', analyzeComments);
  themeToggle.addEventListener('click', toggleTheme);
  copyBtn.addEventListener('click', copyResults);
  
  // Filtres
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      currentFilter = e.target.dataset.filter;
      updateFilterButtons();
      renderComments(allPredictions);
    });
  });
}

// Fonction principale d'analyse
async function analyzeComments() {
  try {
    showLoading();
    hideError();
    
    // 1. Extraire les commentaires de la page YouTube
    const comments = await extractCommentsFromPage();
    
    if (!comments || comments.length === 0) {
      throw new Error('Aucun commentaire trouvé sur cette page. Assurez-vous d\'être sur une vidéo YouTube avec des commentaires.');
    }
    
    console.log(`${comments.length} commentaires extraits`);
    
    // 2. Envoyer à l'API pour analyse
    const results = await sendToAPI(comments);
    
    // 3. Afficher les résultats
    displayResults(results);
    
  } catch (error) {
    showError(error.message);
  } finally {
    hideLoading();
  }
}

// Extraction des commentaires depuis la page YouTube
function extractCommentsFromPage() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]) {
        reject(new Error('Aucun onglet actif trouvé'));
        return;
      }
      
      // Vérifier si on est sur YouTube
      if (!tabs[0].url.includes('youtube.com/watch')) {
        reject(new Error('Veuillez ouvrir une vidéo YouTube'));
        return;
      }
      
      // Envoyer un message au content script
      chrome.tabs.sendMessage(
        tabs[0].id,
        { action: 'extractComments' },
        (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error('Erreur de communication avec la page. Rechargez la page et réessayez.'));
            return;
          }
          
          if (response && response.success) {
            resolve(response.comments);
          } else {
            reject(new Error(response?.error || 'Erreur lors de l\'extraction des commentaires'));
          }
        }
      );
    });
  });
}

// Envoi à l'API pour analyse
async function sendToAPI(comments) {
  try {
    const response = await fetch(`${API_URL}/predict_batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        comments: comments
      })
    });
    
    if (!response.ok) {
      throw new Error(`Erreur API: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
    
  } catch (error) {
    throw new Error(`Impossible de contacter l'API: ${error.message}`);
  }
}

// Affichage des résultats
function displayResults(data) {
  allPredictions = data.predictions;
  
  // Afficher les statistiques
  updateStatistics(data.statistics);
  
  // Afficher les commentaires
  renderComments(allPredictions);
  
  // Afficher les sections
  statsSection.classList.remove('hidden');
  filtersSection.classList.remove('hidden');
  commentsSection.classList.remove('hidden');
}

// Mise à jour des statistiques
function updateStatistics(stats) {
  document.getElementById('totalComments').textContent = stats.total_comments;
  document.getElementById('positivePercent').textContent = `${stats.sentiment_percentages.positive}%`;
  document.getElementById('neutralPercent').textContent = `${stats.sentiment_percentages.neutral}%`;
  document.getElementById('negativePercent').textContent = `${stats.sentiment_percentages.negative}%`;
  
  // Barre de progression
  document.getElementById('positiveBar').style.width = `${stats.sentiment_percentages.positive}%`;
  document.getElementById('neutralBar').style.width = `${stats.sentiment_percentages.neutral}%`;
  document.getElementById('negativeBar').style.width = `${stats.sentiment_percentages.negative}%`;
}

// Affichage des commentaires
function renderComments(predictions) {
  const commentsList = document.getElementById('commentsList');
  commentsList.innerHTML = '';
  
  // Filtrer selon le filtre actif
  const filtered = predictions.filter(p => {
    if (currentFilter === 'all') return true;
    return p.sentiment === currentFilter;
  });
  
  if (filtered.length === 0) {
    commentsList.innerHTML = '<p style="text-align:center; color: var(--text-secondary);">Aucun commentaire dans cette catégorie</p>';
    return;
  }
  
  filtered.forEach(prediction => {
    const item = createCommentItem(prediction);
    commentsList.appendChild(item);
  });
}

// Création d'un élément commentaire
function createCommentItem(prediction) {
  const div = document.createElement('div');
  div.className = `comment-item ${prediction.sentiment}`;
  
  const sentimentText = {
    'positive': ' Positif',
    'neutral': ' Neutre',
    'negative': ' Négatif'
  }[prediction.sentiment];
  
  div.innerHTML = `
    <div class="comment-header">
      <span class="sentiment-badge ${prediction.sentiment}">${sentimentText}</span>
      <span class="confidence">${(prediction.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="comment-text">${escapeHtml(prediction.text)}</div>
  `;
  
  return div;
}

// Mise à jour des boutons de filtre
function updateFilterButtons() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    if (btn.dataset.filter === currentFilter) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

// Copie des résultats
function copyResults() {
  const text = allPredictions.map(p => 
    `[${p.sentiment.toUpperCase()}] ${p.text}`
  ).join('\n\n');
  
  navigator.clipboard.writeText(text).then(() => {
    copyBtn.textContent = ' Copié !';
    setTimeout(() => {
      copyBtn.textContent = ' Copier';
    }, 2000);
  });
}

// Gestion du thème
function toggleTheme() {
  const isDark = document.body.classList.toggle('dark-mode');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  themeToggle.textContent = isDark ? '☀️' : '🌙';
}

function loadTheme() {
  const theme = localStorage.getItem('theme') || 'light';
  if (theme === 'dark') {
    document.body.classList.add('dark-mode');
    themeToggle.textContent = '☀️';
  }
}

// Gestion des états UI
function showLoading() {
  loading.classList.remove('hidden');
  analyzeBtn.disabled = true;
  statsSection.classList.add('hidden');
  filtersSection.classList.add('hidden');
  commentsSection.classList.add('hidden');
}

function hideLoading() {
  loading.classList.add('hidden');
  analyzeBtn.disabled = false;
}

function showError(message) {
  errorSection.classList.remove('hidden');
  document.getElementById('errorMessage').textContent = message;
}

function hideError() {
  errorSection.classList.add('hidden');
}

// Utilitaires
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
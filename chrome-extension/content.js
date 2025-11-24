// content.js - Script qui s'exécute sur les pages YouTube

// Fonction pour extraire les commentaires de la page
function extractComments() {
  const comments = [];
  
  // Sélecteurs pour les commentaires YouTube
  const commentElements = document.querySelectorAll('#content-text');
  
  commentElements.forEach((element) => {
    const text = element.textContent.trim();
    if (text && text.length > 0) {
      comments.push(text);
    }
  });
  
  return comments;
}

// Écouter les messages du popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'extractComments') {
    try {
      const comments = extractComments();
      sendResponse({ 
        success: true, 
        comments: comments,
        count: comments.length 
      });
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
  }
  return true; // Indique qu'on va répondre de manière asynchrone
});

console.log('YouTube Sentiment Analyzer - Content script chargé');
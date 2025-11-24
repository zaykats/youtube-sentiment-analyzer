# YouTube Sentiment Analyzer 

Système MLOps complet pour analyser le sentiment des commentaires YouTube en temps réel avec une précision de **89.12%**.

##  Fonctionnalités

- **Modèle ML haute performance** : TF-IDF + Logistic Regression optimisé avec GridSearchCV
- **API REST** : FastAPI déployée sur Hugging Face Spaces
- **Extension Chrome** : Analyse en temps réel des commentaires YouTube
- **Pipeline MLOps complet** : De la collecte de données au déploiement

## Performance du Modèle

### Métriques Globales
- **Accuracy** : **89.12%** 
- **F1-Score (weighted)** : **0.8902** 
- **Dataset** : 36,982 commentaires Reddit
- **Train/Test Split** : 29,585 / 7,397 (80/20)

### Performance par Classe

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Négatif (-1) | 0.85 | 0.79 | 0.82 | 1,656 |
| Neutre (0) | 0.89 | 0.96 | 0.92 | 2,575 |
| Positif (1) | 0.91 | 0.89 | 0.90 | 3,166 |

### Optimisation
- **Algorithme** : GridSearchCV (5-fold cross-validation)
- **Meilleurs hyperparamètres** :
  - `C`: 10.0
  - `solver`: liblinear
  - `max_iter`: 200
- **Score CV** : 0.8794

### Temps d'Inférence
- **Batch de 50 commentaires** : < 1ms 
- **Temps moyen par commentaire** : < 0.02ms

##  Structure du Projet

```
youtube-sentiment-analyzer/
├── data/
│   ├── raw/                    # Données brutes (reddit.csv)
│   └── processed/              # Données nettoyées (train.csv, test.csv)
├── models/
│   ├── sentiment_model.joblib # Modèle entraîné
│   └── vectorizer.joblib      # Vectoriseur TF-IDF
├── src/
│   ├── data/
│   │   ├── download_data.py   # Téléchargement du dataset
│   │   └── clean_data.py      # Nettoyage et preprocessing
│   ├── models/
│   │   └── train_model.py     # Entraînement et optimisation
│   └── api/
│       └── app.py             # API FastAPI
├── chrome-extension/           # Extension Chrome
├── logs/
│   └── confusion_matrix.png   # Visualisation des performances
├── tests/                      # Tests unitaires
├── requirements.txt            # Dépendances Python
├── Dockerfile                  # Configuration Docker
└── README.md                   # Documentation
```

##  Installation

### Prérequis
- Python 3.10+
- Git
- Compte Hugging Face (pour le déploiement)
- Google Chrome (pour l'extension)

### Étapes d'installation

```bash
# 1. Cloner le repository
git clone https://github.com/zaykats/youtube-sentiment-analyzer
cd youtube-sentiment-analyzer

# 2. Créer l'environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

##  Utilisation

### 1️ Télécharger et Préparer les Données

```bash
# Télécharger le dataset Reddit Sentiment
python src/data/download_data.py

# Nettoyer et préparer les données
python src/data/clean_data.py
```

**Output attendu** :
- `data/raw/reddit.csv` : Dataset brut (36,982 commentaires)
- `data/processed/train.csv` : Données d'entraînement (29,585)
- `data/processed/test.csv` : Données de test (7,397)

### 2️ Entraîner le Modèle

```bash
python src/models/train_model.py
```

**Output attendu** :
```
 Accuracy : 0.8912 (89.12%)
 F1-Score : 0.8902
 Modèle sauvegardé : models/sentiment_model.joblib
 Matrice de confusion : logs/confusion_matrix.png
```

### 3️ Lancer l'API Localement

```bash
# Démarrer l'API FastAPI
python src/api/app.py

# L'API sera disponible sur http://localhost:8000
```

**Endpoints disponibles** :
- `GET /` : Informations sur l'API
- `GET /health` : Vérification de l'état
- `POST /predict_batch` : Analyse de sentiment par batch

### 4️ Tester l'API

```bash
# Test de santé
curl http://localhost:8000/health

# Test de prédiction
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d '{
       "comments": [
         "This is amazing! I love it!",
         "This is terrible, waste of time",
         "It's okay, nothing special"
       ]
     }'
```

**Réponse attendue** :
```json
{
  "predictions": [
    {
      "comment": "This is amazing! I love it!",
      "sentiment": "positive",
      "sentiment_label": 1,
      "confidence": 0.95
    },
    ...
  ],
  "statistics": {
    "total_comments": 3,
    "positive_count": 1,
    "neutral_count": 1,
    "negative_count": 1,
    "positive_percentage": 33.33,
    "neutral_percentage": 33.33,
    "negative_percentage": 33.33,
    "average_confidence": 0.89
  },
  "processing_time": 0.023
}
```

##  Déploiement Docker

### Build l'image Docker

```bash
docker build -t youtube-sentiment-api .
```

### Lancer le container

```bash
docker run -p 7860:7860 youtube-sentiment-api
```

### Déployer sur Hugging Face Spaces

1. Créez un Space sur [huggingface.co/spaces](https://huggingface.co/spaces)
2. Sélectionnez **Docker** comme SDK
3. Clonez votre Space localement
4. Copiez les fichiers nécessaires :
   - `src/api/app.py` → `app_api.py`
   - `models/` → `models/`
   - `Dockerfile`
   - `requirements.txt`
5. Poussez vers Hugging Face

```bash
git push
```

##  Tests et Validation

### Tests Unitaires

```bash
# Exécuter tous les tests
pytest tests/

# Tests avec couverture
pytest --cov=src tests/
```

### Tests de Performance

```bash
# Test de charge API
python tests/load_test.py
```

##  Analyse des Résultats

### Matrice de Confusion

La matrice de confusion montre la répartition des prédictions :

```
Vrai\Prédit  Négatif  Neutre  Positif
Négatif       1307     130     219
Neutre          56    2461      58
Positif        171     171    2824
```

**Interprétation** :
-  Le modèle excelle dans la détection des commentaires **neutres** (96% recall)
-  Bonne performance sur les commentaires **positifs** (89% recall)
-  Légère confusion entre négatifs et positifs (10-15% d'erreur croisée)

### Points Forts
1. **Équilibre** : Bonne performance sur les 3 classes
2. **Rapidité** : Inférence ultra-rapide (< 1ms pour 50 commentaires)
3. **Robustesse** : F1-Score > 0.82 pour toutes les classes

### Améliorations Futures
- [ ] Utiliser des embeddings pré-entraînés (Word2Vec, BERT)
- [ ] Augmenter le dataset avec des commentaires YouTube réels
- [ ] Implémenter un système de re-entraînement continu
- [ ] Ajouter la détection de sarcasme et d'ironie

##  Technologies Utilisées

### Machine Learning
- **scikit-learn** : Modèle et vectorisation
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques

### API & Backend
- **FastAPI** : Framework web moderne
- **uvicorn** : Serveur ASGI
- **pydantic** : Validation de données

### DevOps & Déploiement
- **Docker** : Containerisation
- **Hugging Face Spaces** : Hébergement cloud
- **Git** : Version control

### Frontend
- **Chrome Extension API** : Intégration navigateur
- **JavaScript** : Logique frontend
- **HTML/CSS** : Interface utilisateur

### Architecture du Modèle

```
Input Text → TF-IDF Vectorizer → Logistic Regression → Sentiment Prediction
              (5000 features)      (optimized params)     (-1, 0, 1)
```

**Vectoriseur TF-IDF** :
- `max_features`: 5000
- `ngram_range`: (1, 2) - unigrammes et bigrammes
- `min_df`: 2 - terme doit apparaître dans au moins 2 documents
- `max_df`: 0.9 - ignore termes trop fréquents

**Logistic Regression** :
- `C`: 10.0 - régularisation inverse
- `solver`: liblinear - optimisé pour petits datasets
- `max_iter`: 200 - nombre d'itérations

##  Projet Académique

**Institution** : École Nationale Supérieure d'Arts et Métiers (ENSAM) - Rabat  
**Filière** : INDIA  
**Module** : Virtualisation & Cloud Computing  
**Année Universitaire** : 2025/26

## 👨‍💻 Auteur

Zaykats

##  Remerciements

- Dataset Reddit Sentiment : [Himanshu-1703](https://github.com/Himanshu-1703/reddit-sentiment-analysis)
- FastAPI Documentation
- scikit-learn Community
- Hugging Face Spaces

---

⭐ **Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**

 **Questions ou suggestions ?** Ouvrez une issue sur GitHub !
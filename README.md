
# YouTube Sentiment Analyzer 

Système MLOps complet pour analyser le sentiment des commentaires YouTube en temps réel.

##  Fonctionnalités

- Modèle ML de classification de sentiment (TF-IDF + Logistic Regression)
- API REST FastAPI déployée sur Hugging Face
- Extension Chrome pour analyse en temps réel
- Accuracy > 80%

##  Structure du projet
```
youtube-sentiment-analyzer/
├── data/                  # Données raw et processed
├── models/                # Modèles entraînés
├── src/                   # Code source
│   ├── data/             # Scripts de données
│   ├── models/           # Scripts ML
│   └── api/              # API FastAPI
├── chrome-extension/      # Extension Chrome
├── logs/                  # Logs
└── tests/                 # Tests unitaires
```

##  Installation
```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

##  Utilisation

### 1. Télécharger et préparer les données
```bash
python src/data/download_data.py
python src/data/clean_data.py
```

### 2. Entraîner le modèle
```bash
python src/models/train_model.py
```

### 3. Lancer l'API localement
```bash
python src/api/app.py
# API disponible sur http://localhost:8000
```

##  Performance du modèle

- **Accuracy** : 85%+
- **F1-Score** : 0.80+
- **Temps d'inférence** : <100ms pour 50 commentaires


## 👨‍💻 Auteur

Zaykats


# Image de base légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements_prod.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copier les fichiers nécessaires
COPY app_api.py .
COPY models/ models/

# Exposer le port 7860 (standard Hugging Face)
EXPOSE 7860

# Commande de démarrage
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_data(path='data/processed/reddit_clean.csv'):
    """Charge les données prétraitées"""
    print("📂 Chargement des données...")
    df = pd.read_csv(path)
    print(f"✅ {len(df)} commentaires chargés")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Divise les données en train/test"""
    print(f"\n📊 Division des données (test_size={test_size})...")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Train set : {len(X_train)} commentaires")
    print(f"✅ Test set : {len(X_test)} commentaires")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, optimize=True):
    """Entraîne le modèle avec TF-IDF + Logistic Regression"""
    
    print("\n🔧 Création du vectoriseur TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        strip_accents='unicode'
    )
    
    print("🔄 Transformation des textes en vecteurs TF-IDF...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"✅ Matrice TF-IDF : {X_train_tfidf.shape}")
    
    if optimize:
        print("\n🎯 Optimisation des hyperparamètres avec GridSearchCV...")
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [200, 500]
        }
        
        model = LogisticRegression(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_tfidf, y_train)
        
        print(f"\n✅ Meilleurs paramètres : {grid_search.best_params_}")
        print(f"✅ Meilleur score F1 (CV) : {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        print("\n🚀 Entraînement du modèle Logistic Regression...")
        model = LogisticRegression(
            C=1.0, 
            solver='liblinear', 
            max_iter=500, 
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_tfidf, y_train)
    
    print("✅ Modèle entraîné avec succès !")
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Évalue le modèle sur le test set"""
    
    print("\n📈 ÉVALUATION DU MODÈLE")
    print("="*60)
    
    # Transformation
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Prédictions
    y_pred = model.predict(X_test_tfidf)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 F1-Score (weighted) : {f1_weighted:.4f}")
    
    print("\n📊 Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred, 
                                target_names=['Négatif (-1)', 'Neutre (0)', 'Positif (1)']))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔢 Matrice de confusion :")
    print(cm)
    
    # Visualisation
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Négatif', 'Neutre', 'Positif'],
                yticklabels=['Négatif', 'Neutre', 'Positif'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Matrice de confusion sauvegardée : logs/confusion_matrix.png")
    
    # Test de vitesse
    print("\n⏱️  Test de performance (temps d'inférence)...")
    batch_size = 50
    X_batch = X_test[:batch_size]
    X_batch_tfidf = vectorizer.transform(X_batch)
    
    start_time = time.time()
    _ = model.predict(X_batch_tfidf)
    inference_time = (time.time() - start_time) * 1000  # en ms
    
    print(f"✅ Temps d'inférence pour {batch_size} commentaires : {inference_time:.2f} ms")
    print(f"✅ Temps moyen par commentaire : {inference_time/batch_size:.2f} ms")
    
    return accuracy, f1_weighted

def save_model(model, vectorizer, model_path='models/sentiment_model.joblib',
               vectorizer_path='models/vectorizer.joblib'):
    """Sauvegarde le modèle et le vectoriseur"""
    
    print("\n💾 Sauvegarde du modèle...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"✅ Modèle sauvegardé : {model_path}")
    print(f"✅ Vectoriseur sauvegardé : {vectorizer_path}")

def main():
    """Pipeline complet d'entraînement"""
    
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE")
    print("="*60)
    
    # 1. Charger les données
    df = load_data()
    
    # 2. Diviser les données
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 3. Entraîner le modèle
    model, vectorizer = train_model(X_train, y_train, optimize=True)
    
    # 4. Évaluer le modèle
    accuracy, f1 = evaluate_model(model, vectorizer, X_test, y_test)
    
    # 5. Sauvegarder le modèle
    save_model(model, vectorizer)
    
    print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print(f"📊 Accuracy finale : {accuracy:.4f}")
    print(f"📊 F1-Score finale : {f1:.4f}")

if __name__ == "__main__":
    main()
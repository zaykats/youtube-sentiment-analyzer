import pandas as pd
import re
import os

def clean_text(text):
    """
    Nettoie un texte en supprimant URLs, mentions, caractères spéciaux
    """
    if pd.isna(text):
        return ""
    
    # Convertir en string
    text = str(text)
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer les hashtags (garder le texte)
    text = re.sub(r'#', '', text)
    
    # Supprimer les caractères spéciaux (garder lettres, chiffres, espaces, ponctuation basique)
    text = re.sub(r'[^\w\s.,!?\'-]', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    # Mettre en minuscules
    text = text.lower()
    
    return text

def preprocess_dataset(input_path='data/raw/reddit.csv', 
                       output_path='data/processed/reddit_clean.csv'):
    """
    Prétraite le dataset complet
    """
    print("🔧 Chargement du dataset...")
    df = pd.read_csv(input_path)
    
    print(f" Dataset original : {len(df)} lignes")
    
    # Identifier les colonnes de texte et de label
    text_col = 'clean_comment' if 'clean_comment' in df.columns else 'comment'
    label_col = 'category'
    
    print(f"\n Nettoyage de la colonne '{text_col}'...")
    
    # Nettoyer les textes
    df['text'] = df[text_col].apply(clean_text)
    
    # Supprimer les lignes vides après nettoyage
    df = df[df['text'].str.len() > 0]
    print(f" Après suppression des textes vides : {len(df)} lignes")
    
    # Renommer la colonne de label en 'label'
    df['label'] = df[label_col]
    
    # Mapper les labels (-1, 0, 1)
    df['label'] = df['label'].map({-1: -1, 0: 0, 1: 1})
    
    # Supprimer les lignes avec labels manquants
    df = df.dropna(subset=['label'])
    
    # Garder uniquement les colonnes nécessaires
    df_clean = df[['text', 'label']].copy()
    
    # Analyse de la distribution
    print("\n Distribution finale des labels :")
    print(df_clean['label'].value_counts().sort_index())
    print("\nPourcentages :")
    print(df_clean['label'].value_counts(normalize=True).sort_index() * 100)
    
    # Statistiques sur la longueur des textes
    df_clean['text_length'] = df_clean['text'].str.len()
    print("\n Statistiques de longueur des textes :")
    print(df_clean['text_length'].describe())
    
    # Sauvegarder
    os.makedirs('data/processed', exist_ok=True)
    df_clean.drop('text_length', axis=1).to_csv(output_path, index=False)
    print(f"\n Dataset nettoyé sauvegardé : {output_path}")
    print(f" Total final : {len(df_clean)} commentaires")
    
    return df_clean

if __name__ == "__main__":
    df_clean = preprocess_dataset()
    
    print("\n Exemples de commentaires nettoyés :")
    print(df_clean.sample(5))
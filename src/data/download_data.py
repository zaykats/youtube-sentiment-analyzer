import pandas as pd
import requests
import os

def download_reddit_dataset():
    """
    Télécharge le dataset Reddit Sentiment Analysis
    """
    url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    
    print(" Téléchargement du dataset en cours...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Créer le dossier si nécessaire
        os.makedirs('data/raw', exist_ok=True)
        
        # Sauvegarder le fichier
        output_path = 'data/raw/reddit.csv'
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f" Dataset téléchargé avec succès : {output_path}")
        
        # Charger et afficher les statistiques
        df = pd.read_csv(output_path)
        
        print("\n STATISTIQUES DU DATASET")
        print("="*50)
        print(f"Nombre total de commentaires : {len(df)}")
        print(f"Nombre de colonnes : {len(df.columns)}")
        print(f"\nColonnes disponibles : {list(df.columns)}")
        
        # Distribution des labels
        if 'category' in df.columns:
            print("\n Distribution des sentiments :")
            print(df['category'].value_counts().sort_index())
            print("\nPourcentages :")
            print(df['category'].value_counts(normalize=True).sort_index() * 100)
        
        # Informations sur les données manquantes
        print("\n Données manquantes :")
        print(df.isnull().sum())
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f" Erreur lors du téléchargement : {e}")
        return None
    except Exception as e:
        print(f" Erreur inattendue : {e}")
        return None

if __name__ == "__main__":
    df = download_reddit_dataset()
    
    if df is not None:
        print("\n Dataset prêt pour le traitement !")
        print("\nPremiers exemples :")
        print(df.head())
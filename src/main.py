import argparse
from train_model import train_model  # Assure-toi que cette fonction est bien définie

def main(train_path, test_path, model_path):
    # Entraîner le modèle et récupérer les performances
    metrics = train_model(train_path, test_path, model_path)
    
    # Afficher les métriques
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print("Modèle entraîné et enregistré avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Chemin vers le fichier d'entraînement")
    parser.add_argument("--test_path", type=str, required=True, help="Chemin vers le fichier de test")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin pour sauvegarder le modèle")
    
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.model_path)


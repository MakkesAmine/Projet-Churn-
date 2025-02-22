import argparse
import pandas as pd
from model_pipeline import (
    preprocess_data,
    optimizer_hyperparameters,
    train_model,
    evaluate_model,
    save_model
)

def main(train_path, test_path, model_path):
    # Étape 1 : Prétraitement des données
    print("Prétraitement des données...")
    X_train, X_test, y_train, y_test = preprocess_data(train_path, test_path)
    
    # Étape 2 : Optimisation des hyperparamètres
    print("Optimisation des hyperparamètres...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    hyperparameters = optimizer_hyperparameters(X_train, y_train, param_grid)
    
    # Étape 3 : Entraînement du modèle avec les meilleurs hyperparamètres
    print("Entraînement du modèle...")
    model = train_model(X_train, y_train, hyperparameters)
    
    # Étape 4 : Évaluation du modèle
    print("Évaluation du modèle...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Affichage des métriques
    print("\nMétriques du modèle :")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Étape 5 : Sauvegarde du modèle
    print("\nSauvegarde du modèle...")
    save_model(model, model_path)
    print(f"Modèle entraîné et enregistré avec succès à l'emplacement : {model_path}")

if __name__ == "__main__":
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Chemin vers le fichier d'entraînement")
    parser.add_argument("--test_path", type=str, required=True, help="Chemin vers le fichier de test")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin pour sauvegarder le modèle")
    
    # Analyse des arguments
    args = parser.parse_args()
    
    # Exécution de la fonction principale
    main(args.train_path, args.test_path, args.model_path)

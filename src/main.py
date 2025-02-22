import argparse
from pathlib import Path
import mlflow
from src.model_pipeline import (
    preprocess_data,
    optimizer_hyperparameters,
    train_model,
    evaluate_model,
    save_model
)

def main(train_path, test_path, model_path):
    mlflow.set_experiment("Churn Prediction")

    # Prétraitement des données
    X_train, X_test, y_train, y_test = preprocess_data(train_path, test_path)

    # Optimisation des hyperparamètres
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
    best_params = optimizer_hyperparameters(X_train, y_train, param_grid)

    # Entraînement du modèle
    model = train_model(X_train, y_train, best_params)

    # Évaluation du modèle
    metrics = evaluate_model(model, X_test, y_test)
    print("Performance du modèle :", metrics)

    # Sauvegarde du modèle
    save_model(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute le pipeline de prédiction du churn.")
    parser.add_argument("--train_path", type=str, required=True, help="Chemin du fichier d'entraînement")
    parser.add_argument("--test_path", type=str, required=True, help="Chemin du fichier de test")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin où enregistrer le modèle")

    args = parser.parse_args()
    
    main(args.train_path, args.test_path, args.model_path)


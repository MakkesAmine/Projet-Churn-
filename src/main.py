import argparse
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Nom de l'expérience et du modèle
EXPERIMENT_NAME = "churn_prediction"
MODEL_NAME = "Churn_Prediction_Model"

# Configuration de MLflow 
mlflow.set_tracking_uri("file:./mlruns")  # Utilisation d'un répertoire local
print(f"Current tracking URI: {mlflow.get_tracking_uri()}")

# Vérifier si l'expérience existe, sinon la créer
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Création de l'expérience '{EXPERIMENT_NAME}'.")
mlflow.set_experiment(EXPERIMENT_NAME)

def transition_model_stage(model_name, model_version, stage):
    """ Change le stage du modèle (Staging, Production, Archived) """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage
    )
    logging.info(f"Modèle {model_name} (version {model_version}) est maintenant en '{stage}' !")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Machine Learning pour la prédiction du churn.")

    parser.add_argument("--prepare", action="store_true", help="Préparer les données.")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle.")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle.")
    parser.add_argument("--save", type=str, help="Sauvegarder le modèle dans un fichier.")
    parser.add_argument("--load", type=str, help="Charger un modèle existant.")
    parser.add_argument("--train_path", type=str, help="Chemin du fichier CSV d'entraînement.")
    parser.add_argument("--test_path", type=str, help="Chemin du fichier CSV de test.")
    parser.add_argument("--stage", type=str, choices=["Staging", "Production", "Archived"], help="Changer le stage d'un modèle existant.")
    parser.add_argument("--model_version", type=str, help="Version du modèle à promouvoir.")
    
    args = parser.parse_args()
    
    # Gestion du staging sans entraînement
    if args.stage and args.model_version:
        logging.info(f"Changement du modèle {MODEL_NAME} (version {args.model_version}) vers '{args.stage}'...")
        transition_model_stage(MODEL_NAME, args.model_version, args.stage)
        print(f"Modèle {MODEL_NAME} (version {args.model_version}) est maintenant en '{args.stage}' !")
        return  # Sortie immédiate après mise à jour du stage
    
    if not (args.train_path and args.test_path):
        parser.error("Les arguments --train_path et --test_path sont requis sauf si vous changez uniquement le stage du modèle.")
    
    logging.info("Chargement et préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
    
    model = None
    
    with mlflow.start_run():
        if args.load:
            logging.info(f"Chargement du modèle depuis {args.load}...")
            model = load_model(args.load)
        elif args.train:
            logging.info("Entraînement du modèle...")
            model, params = train_model(X_train, y_train)

            if model:
                logging.info(f"Modèle entraîné avec succès. Paramètres : {params}")
                for param, value in params.items():
                    mlflow.log_param(param, value)
                
                mlflow.sklearn.log_model(model, "churn_model")
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/churn_model"
                registered_model = mlflow.register_model(model_uri, MODEL_NAME)
                logging.info(f"Modèle '{MODEL_NAME}' enregistré avec la version {registered_model.version}.")
                
                if args.stage:
                    transition_model_stage(MODEL_NAME, registered_model.version, args.stage)

                accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
                result_message = (
                    f"Résultats de l'entraînement :\n"
                    f"- Accuracy: {accuracy:.4f}\n"
                    f"- Precision: {precision:.4f}\n"
                    f"- Recall: {recall:.4f}\n"
                    f"- F1-score: {f1:.4f}"
                )
                logging.info(result_message)
                print(result_message)
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

        if model and args.evaluate:
            logging.info("Évaluation du modèle...")
            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            result_message = (
                f"Résultats de l'évaluation :\n"
                f"- Accuracy: {accuracy:.4f}\n"
                f"- Precision: {precision:.4f}\n"
                f"- Recall: {recall:.4f}\n"
                f"- F1-score: {f1:.4f}"
            )
            logging.info(result_message)
            print(result_message)

        if model and args.save:
            logging.info(f"Sauvegarde du modèle dans {args.save}...")
            save_model(model, args.save)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Erreur dans le pipeline : {str(e)}")
    finally:
        mlflow.end_run()


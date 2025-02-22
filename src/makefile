# Définition des variables
PYTHON=python3
VENV=venv
REQUIREMENTS=requirements.txt
TRAIN_PATH=churn-bigml-80.csv
TEST_PATH=churn-bigml-20.csv

# Variables Docker
IMAGE_NAME=aminemakkes/ml_project
TAG=latest

.PHONY: install data train evaluate test save start-api test-api notebook 
#build run stop push clean

# Création de l'environnement virtuel et installation des dépendances
install:
	@echo "📦 Création de l'environnement virtuel et installation des dépendances..."
	@${PYTHON} -m venv ${VENV}
	@. ${VENV}/bin/activate && pip install --upgrade pip && pip install -r ${REQUIREMENTS}

# Préparation des données
data:
	@echo "📊 Préparation des données..."
	@. ${VENV}/bin/activate && ${PYTHON} main.py --prepare --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Entraînement du modèle
train:
	@echo "🤖 Entraînement du modèle..."
	@. ${VENV}/bin/activate && ${PYTHON} main.py --train --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Évaluation du modèle
evaluate:
	@echo "📈 Évaluation du modèle..."
	@. ${VENV}/bin/activate && ${PYTHON} main.py --evaluate --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Exécution des tests
test:
	@echo "🧪 Exécution des tests..."
	@. ${VENV}/bin/activate && pytest test_environment.py

# Sauvegarde du modèle
save:
	@echo "💾 Sauvegarde du modèle..."
	@. ${VENV}/bin/activate && ${PYTHON} main.py --save churn_model.pkl --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Démarrage de l'API FastAPI
start-api:
	@echo "🚀 Démarrage de l'API FastAPI..."
	@. ${VENV}/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
	@sleep 3
	@echo "🌐 Ouverture de Swagger..."

mlflow-ui:
	@echo "🌐 Lancement de l'interface MLflow sur http..."
	@. ${VENV}/bin/activate && mlflow ui --host 0.0.0.0 --port 5000

# Nettoyage des fichiers temporaires
clean:
	@echo "🧹 Nettoyage des fichiers temporaires..."
	@rm -rf _pycache_ *.pyc *.pyo *.log ${VENV}

# Lancer Jupyter Notebook pour l'exploration
notebook:
	@echo "📓 Lancement de Jupyter Notebook..."
	@. ${VENV}/bin/activate && jupyter notebook

# 🐳 Docker Commands

## Construire l'image Docker
build:
	@echo "🔨 Construction de l'image Docker..."
	sudo docker build -t $(IMAGE_NAME):$(TAG) .

## Lancer le conteneur
run:
	@echo "🚀 Lancement du conteneur..."
	sudo docker run -d -p 8000:8000 --name fastapi-app $(IMAGE_NAME):$(TAG)

## Stopper et supprimer le conteneur
stop:
	@echo "🛑 Arrêt du conteneur..."
	sudo docker stop fastapi-app && sudo docker rm fastapi-app

## Pousser l’image sur Docker Hub
push:
	@echo "📤 Poussée de l'image sur Docker Hub..."
	sudo docker push $(IMAGE_NAME):$(TAG)

## Nettoyer les images et conteneurs inutiles
docker-clean:
	@echo "🧹 Nettoyage des images et conteneurs Docker..."
	sudo docker system prune -f

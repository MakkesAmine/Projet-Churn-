"""
Pipeline de machine learning pour la prédiction de churn.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.base import BaseEstimator
import joblib
import mlflow
from scipy.stats import zscore

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.sklearn.autolog()

def preprocess_data(train_path: Union[str, Path],
                    test_path: Union[str, Path],
                    outlier_threshold: float = 3.0) -> Tuple[pd.DataFrame, ...]:
    """
    Prépare et nettoie les données pour l'entraînement.

    Args:
        train_path: Chemin vers les données d'entraînement
        test_path: Chemin vers les données de test
        outlier_threshold: Seuil pour la détection des outliers

    Returns:
        Tuple contenant les données préparées (X_train, X_test, y_train, y_test)
    """
    # Chargement des données
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Suppression des colonnes non informatives
    columns_to_drop = [
        'State', 'Area code', 'Total day minutes',
        'Total eve minutes', 'Total night minutes', 'Total intl minutes'
    ]
    train_clean = train_df.drop(columns=columns_to_drop)
    test_clean = test_df.drop(columns=columns_to_drop)

    # Encodage des caractéristiques catégorielles
    categorical_features = ['International plan', 'Voice mail plan']
    encoders = {col: LabelEncoder() for col in categorical_features}
    
    for col in categorical_features:
        train_clean[col] = encoders[col].fit_transform(train_clean[col])
        test_clean[col] = encoders[col].transform(test_clean[col])

    # Encodage de la cible
    target_encoder = LabelEncoder()
    train_clean['Churn'] = target_encoder.fit_transform(train_clean['Churn'])
    test_clean['Churn'] = target_encoder.transform(test_clean['Churn'])

    # Suppression des outliers (seulement sur le train)
    numerical_features = [
        'Account length', 'Number vmail messages', 'Total day calls',
        'Total day charge', 'Total eve calls', 'Total eve charge',
        'Total night calls', 'Total night charge', 'Total intl calls',
        'Total intl charge', 'Customer service calls'
    ]
    train_clean = _remove_outliers(train_clean, numerical_features, threshold=outlier_threshold)

    # Normalisation des données
    scaler = MinMaxScaler()
    train_clean[numerical_features] = scaler.fit_transform(train_clean[numerical_features])
    test_clean[numerical_features] = scaler.transform(test_clean[numerical_features])

    # Séparation features/target
    X_train = train_clean.drop('Churn', axis=1)
    y_train = train_clean['Churn']
    X_test = test_clean.drop('Churn', axis=1)
    y_test = test_clean['Churn']

    return X_train, X_test, y_train, y_test

def _remove_outliers(data: pd.DataFrame,
                    numerical_features: list,
                    method: str = "iqr",
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Supprime les outliers des données numériques.

    Args:
        data: DataFrame contenant les données
        numerical_features: Liste des colonnes numériques
        method: Méthode de détection (zscore/iqr)
        threshold: Seuil de détection

    Returns:
        DataFrame filtré
    """
    if method == "zscore":
        z_scores = np.abs(zscore(data[numerical_features]))
        return data[(z_scores < threshold).all(axis=1)]
    
    if method == "iqr":
        masks = []
        for col in numerical_features:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            mask = data[col].between(q1 - threshold*iqr, q3 + threshold*iqr)
            masks.append(mask)
        return data[np.all(masks, axis=0)]
    
    raise ValueError(f"Méthode {method} non supportée")

def optimizer_hyperparameters(X: pd.DataFrame,
                             y: pd.Series,
                             param_grid: Dict[str, list]) -> Dict[str, Any]:
    """
    Optimise les hyperparamètres du modèle.

    Args:
        X: Données d'entraînement
        y: Cible d'entraînement
        param_grid: Grille d'hyperparamètres

    Returns:
        Meilleurs paramètres trouvés
    """
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        mlflow.log_params(grid_search.best_params_)
        return grid_search.best_params_

def train_model(X: pd.DataFrame,
               y: pd.Series,
               hyperparameters: Dict[str, Any] = None) -> BaseEstimator:
    """
    Entraîne un modèle avec les hyperparamètres spécifiés.

    Args:
        X: Données d'entraînement
        y: Cible d'entraînement
        hyperparameters: Hyperparamètres du modèle

    Returns:
        Modèle entraîné
    """
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(
            **(hyperparameters or {}),
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X, y)
        return model

def evaluate_model(model: BaseEstimator,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> Dict[str, float]:
    """
    Évalue les performances du modèle.

    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Cible de test

    Returns:
        Dictionnaire des métriques
    """
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print(classification_report(y_test, y_pred))
    mlflow.log_metrics(metrics)
    return metrics

def save_model(model: BaseEstimator, path: Union[str, Path]) -> None:
    """
    Sauvegarde le modèle entraîné.

    Args:
        model: Modèle à sauvegarder
        path: Chemin de sauvegarde
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Union[str, Path]) -> BaseEstimator:
    """
    Charge un modèle sauvegardé.

    Args:
        path: Chemin vers le modèle

    Returns:
        Modèle chargé
    """
    return joblib.load(path)

import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ..src.model_pipeline import (  # Import relatif
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# Chemins des fichiers de test
TRAIN_PATH = "data/churn-bigml-80.csv"
TEST_PATH = "data/churn-bigml-20.csv"

# Tests pour la fonction preprocess_data
def test_preprocess_data():
    """
    Teste la fonction preprocess_data.
    Vérifie que les données sont correctement prétraitées.
    """
    X_train, X_test, y_train, y_test = preprocess_data(TRAIN_PATH, TEST_PATH)

    # Vérifier que les données sont des DataFrames/Series
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Vérifier que les données ne sont pas vides
    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty

    # Vérifier que les colonnes sont correctes
    expected_columns = [
        'Account length', 'Number vmail messages', 'Total day calls',
        'Total day charge', 'Total eve calls', 'Total eve charge',
        'Total night calls', 'Total night charge', 'Total intl calls',
        'Total intl charge', 'Customer service calls',
        'International plan', 'Voice mail plan'
    ]
    assert all(col in X_train.columns for col in expected_columns)

# Tests pour la fonction train_model
def test_train_model():
    """
    Teste la fonction train_model.
    Vérifie que le modèle est correctement entraîné.
    """
    X_train, X_test, y_train, y_test = preprocess_data(TRAIN_PATH, TEST_PATH)
    model, params = train_model(X_train, y_train)

    # Vérifier que le modèle est bien entraîné
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "feature_importances_")

# Tests pour la fonction evaluate_model
def test_evaluate_model():
    """
    Teste la fonction evaluate_model.
    Vérifie que les métriques sont correctement calculées.
    """
    X_train, X_test, y_train, y_test = preprocess_data(TRAIN_PATH, TEST_PATH)
    model, _ = train_model(X_train, y_train)
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

    # Vérifier que les métriques sont des nombres valides
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

# Tests pour les fonctions save_model et load_model
def test_save_and_load_model(tmpdir):
    """
    Teste les fonctions save_model et load_model.
    Vérifie que le modèle peut être sauvegardé et chargé correctement.
    """
    X_train, _, y_train, _ = preprocess_data(TRAIN_PATH, TEST_PATH)
    model, _ = train_model(X_train, y_train)

    # Sauvegarder le modèle
    model_path = tmpdir.join("test_model.joblib")
    save_model(model, model_path)

    # Charger le modèle
    loaded_model = load_model(model_path)

    # Vérifier que le modèle chargé est identique au modèle original
    assert isinstance(loaded_model, RandomForestClassifier)
    assert loaded_model.get_params() == model.get_params()

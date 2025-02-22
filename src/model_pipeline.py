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
from scipy.stats import zscore

def preprocess_data(train_path: Union[str, Path],
                    test_path: Union[str, Path],
                    outlier_threshold: float = 3.0) -> Tuple[pd.DataFrame, ...]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    columns_to_drop = [
        'State', 'Area code', 'Total day minutes',
        'Total eve minutes', 'Total night minutes', 'Total intl minutes'
    ]
    train_clean = train_df.drop(columns=columns_to_drop)
    test_clean = test_df.drop(columns=columns_to_drop)
    categorical_features = ['International plan', 'Voice mail plan']
    encoders = {col: LabelEncoder() for col in categorical_features}
    
    for col in categorical_features:
        train_clean[col] = encoders[col].fit_transform(train_clean[col])
        test_clean[col] = encoders[col].transform(test_clean[col])
    
    target_encoder = LabelEncoder()
    train_clean['Churn'] = target_encoder.fit_transform(train_clean['Churn'])
    test_clean['Churn'] = target_encoder.transform(test_clean['Churn'])
    numerical_features = [
        'Account length', 'Number vmail messages', 'Total day calls',
        'Total day charge', 'Total eve calls', 'Total eve charge',
        'Total night calls', 'Total night charge', 'Total intl calls',
        'Total intl charge', 'Customer service calls'
    ]
    train_clean = _remove_outliers(train_clean, numerical_features, threshold=outlier_threshold)
    scaler = MinMaxScaler()
    train_clean[numerical_features] = scaler.fit_transform(train_clean[numerical_features])
    test_clean[numerical_features] = scaler.transform(test_clean[numerical_features])
    X_train = train_clean.drop('Churn', axis=1)
    y_train = train_clean['Churn']
    X_test = test_clean.drop('Churn', axis=1)
    y_test = test_clean['Churn']
    return X_train, X_test, y_train, y_test

def _remove_outliers(data: pd.DataFrame,
                    numerical_features: list,
                    method: str = "iqr",
                    threshold: float = 3.0) -> pd.DataFrame:
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
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    print("Meilleurs hyperparamètres trouvés :", best_params)  # DEBUG
    
    if not isinstance(best_params, dict):
        raise ValueError("optimizer_hyperparameters doit retourner un dictionnaire.")
    
    return best_params

def train_model(X: pd.DataFrame,
               y: pd.Series,
               hyperparameters: Dict[str, Any] = None) -> BaseEstimator:
    if hyperparameters is None:
        hyperparameters = {}
    elif not isinstance(hyperparameters, dict):
        raise ValueError("Les hyperparamètres doivent être un dictionnaire.")

    model = RandomForestClassifier(
        **hyperparameters,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    return model

def evaluate_model(model: BaseEstimator,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    print(classification_report(y_test, y_pred))
    return metrics

def save_model(model: BaseEstimator, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Union[str, Path]) -> BaseEstimator:
    return joblib.load(path)


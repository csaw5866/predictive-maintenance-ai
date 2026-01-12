"""
Machine learning models and training module
"""

from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from pma.config import settings
from pma.logger import logger


class ModelTrainer:
    """Train and evaluate ML models"""

    def __init__(self, experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME):
        """Initialize trainer with MLflow"""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        self.logger = logger

    def train_classification_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Train multiple classification models for failure prediction

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels (0=normal, 1=failure_imminent)
            y_test: Test labels
            feature_names: Feature names for logging

        Returns:
            Dictionary with model names and their metrics
        """
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=settings.RANDOM_STATE
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=settings.RANDOM_STATE,
                n_jobs=settings.N_JOBS,
            ),
            "xgboost": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
                verbosity=0,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
                verbose=-1,
            ),
        }

        results = {}

        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")

            with mlflow.start_run(run_name=model_name):
                # Train
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Metrics
                metrics = {
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
                }

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                metrics["tn"] = int(cm[0, 0])
                metrics["fp"] = int(cm[0, 1])
                metrics["fn"] = int(cm[1, 0])
                metrics["tp"] = int(cm[1, 1])

                # Log to MLflow
                mlflow.log_params({"model_type": model_name})
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model
                mlflow.sklearn.log_model(model, model_name)

                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                }

                self.logger.info(f"  F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

        return results

    def train_regression_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        """
        Train multiple regression models for RUL prediction

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training RUL values
            y_test: Test RUL values

        Returns:
            Dictionary with model names and their metrics
        """
        models = {
            "ridge": Ridge(alpha=1.0, random_state=settings.RANDOM_STATE),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=settings.RANDOM_STATE,
                n_jobs=settings.N_JOBS,
            ),
            "xgboost": XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
                verbosity=0,
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
                verbose=-1,
            ),
        }

        results = {}

        for model_name, model in models.items():
            self.logger.info(f"Training RUL {model_name}...")

            with mlflow.start_run(run_name=f"rul_{model_name}"):
                # Train
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                metrics = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                }

                # Log to MLflow
                mlflow.log_params({"model_type": f"rul_{model_name}"})
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                mlflow.sklearn.log_model(model, f"rul_{model_name}")

                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": y_pred,
                }

                self.logger.info(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return results

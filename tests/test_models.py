"""
Tests for ML models
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score, mean_squared_error
from pma.models import ModelTrainer


class TestModelTrainer:
    """Test model training"""

    def test_classification_training(self, feature_data):
        """Test classification model training"""
        # Prepare data
        sensor_cols = [col for col in feature_data.columns if col.startswith("sensor_")]
        feature_cols = [col for col in feature_data.columns 
                       if col not in sensor_cols and col not in ["machine_id", "cycle"]]

        X = feature_data[feature_cols[:10]].values  # Use first 10 features
        y = np.random.randint(0, 2, len(X))  # Binary labels

        X_train = X[:70]
        X_test = X[70:]
        y_train = y[:70]
        y_test = y[70:]

        # Train
        trainer = ModelTrainer()
        results = trainer.train_classification_models(X_train, X_test, y_train, y_test)

        # Check results
        assert len(results) > 0
        for model_name, result in results.items():
            assert "metrics" in result
            assert "precision" in result["metrics"]
            assert "recall" in result["metrics"]
            assert "f1" in result["metrics"]

    def test_regression_training(self, feature_data):
        """Test regression model training"""
        # Prepare data
        sensor_cols = [col for col in feature_data.columns if col.startswith("sensor_")]
        feature_cols = [col for col in feature_data.columns 
                       if col not in sensor_cols and col not in ["machine_id", "cycle"]]

        X = feature_data[feature_cols[:10]].values
        y = np.random.uniform(50, 300, len(X))  # RUL values

        X_train = X[:70]
        X_test = X[70:]
        y_train = y[:70]
        y_test = y[70:]

        # Train
        trainer = ModelTrainer()
        results = trainer.train_regression_models(X_train, X_test, y_train, y_test)

        # Check results
        assert len(results) > 0
        for model_name, result in results.items():
            assert "metrics" in result
            assert "rmse" in result["metrics"]
            assert "mae" in result["metrics"]

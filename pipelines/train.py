"""
Training pipeline for predictive maintenance models
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pma.config import settings
from pma.data import DataDownloader, DataPreprocessor
from pma.features import FeatureEngineer
from pma.logger import logger
from pma.models import ModelTrainer


def run_training_pipeline():
    """Execute complete training pipeline"""

    logger.info("=" * 80)
    logger.info("PREDICTIVE MAINTENANCE TRAINING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Download and load data
    logger.info("\n[1/5] Loading Dataset...")
    downloader = DataDownloader()
    train_df, test_df, rul_df = downloader.load_data()

    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")

    # Step 2: Preprocess data
    logger.info("\n[2/5] Preprocessing Data...")
    preprocessor = DataPreprocessor()

    # Identify sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith("sensor_")]
    logger.info(f"  Sensors identified: {len(sensor_cols)}")

    # Normalize sensor data
    train_normalized, norm_params = preprocessor.normalize_data(train_df, sensor_cols)
    test_normalized, _ = preprocessor.normalize_data(test_df, sensor_cols)

    # Add RUL labels
    train_df = preprocessor.add_rul_labels(train_normalized, rul_df, threshold_days=settings.RUL_THRESHOLD_DAYS)

    # Step 3: Feature Engineering
    logger.info("\n[3/5] Engineering Features...")
    feature_engineer = FeatureEngineer()
    train_features = feature_engineer.engineer_features(train_df)
    test_features = feature_engineer.engineer_features(test_normalized)

    # Remove original sensor columns for model input
    feature_cols = [col for col in train_features.columns if not col.startswith("sensor_")]
    # Exclude identifiers and label columns
    feature_cols = [
        col
        for col in feature_cols
        if col not in [
            "machine_id",
            "cycle",
            "max_cycles",
            "cycles_to_failure",
            "failure_imminent",
        ]
    ]

    X_train = train_features[feature_cols].values
    y_train_classification = train_features["failure_imminent"].values
    y_train_regression = train_features["cycles_to_failure"].values

    X_test = test_features[feature_cols].values if "failure_imminent" not in test_features.columns else test_features[feature_cols].values

    logger.info(f"  Features generated: {X_train.shape[1]}")
    logger.info(f"  Training samples: {X_train.shape[0]}")

    # Step 4: Train Models
    logger.info("\n[4/5] Training Models...")

    # Split data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train,
        y_train_classification,
        test_size=settings.VALIDATION_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y_train_classification,
    )

    # Train classification models
    trainer = ModelTrainer()
    classification_results = trainer.train_classification_models(
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
        feature_names=feature_cols,
    )

    # Train regression models
    X_train_split_rul, X_val_split_rul, y_train_split_rul, y_val_split_rul = train_test_split(
        X_train,
        y_train_regression,
        test_size=settings.VALIDATION_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    regression_results = trainer.train_regression_models(
        X_train_split_rul,
        X_val_split_rul,
        y_train_split_rul,
        y_val_split_rul,
    )

    # Step 5: Save Models
    logger.info("\n[5/5] Saving Models...")
    import joblib

    models_dir = Path(settings.MODELS_PATH)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save best classification model
    best_clf_model = max(
        classification_results.items(),
        key=lambda x: x[1]["metrics"]["f1"],
    )
    joblib.dump(best_clf_model[1]["model"], models_dir / "best_classifier.pkl")
    logger.info(f"  Saved: {best_clf_model[0]} classifier")
    logger.info(f"    F1 Score: {best_clf_model[1]['metrics']['f1']:.4f}")

    # Save best regression model
    best_reg_model = min(
        regression_results.items(),
        key=lambda x: x[1]["metrics"]["rmse"],
    )
    joblib.dump(best_reg_model[1]["model"], models_dir / "best_regressor.pkl")
    logger.info(f"  Saved: {best_reg_model[0]} regressor")
    logger.info(f"    RMSE: {best_reg_model[1]['metrics']['rmse']:.4f}")

    # Save feature names
    feature_dict = {"feature_names": feature_cols, "n_features": len(feature_cols)}
    import json

    with open(models_dir / "features.json", "w") as f:
        json.dump(feature_dict, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)

    return {
        "classification_results": classification_results,
        "regression_results": regression_results,
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    run_training_pipeline()

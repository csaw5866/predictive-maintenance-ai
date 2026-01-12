"""
Enhanced training pipeline with monitoring and evaluation
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pma.anomaly import AnomalyDetector
from pma.config import settings
from pma.data import DataDownloader, DataPreprocessor
from pma.features import FeatureEngineer
from pma.logger import logger
from pma.models import ModelTrainer
from pma.utils import DataLogger, ModelManager, PerformanceMonitor


def run_complete_pipeline():
    """Execute complete training and evaluation pipeline"""

    monitor = PerformanceMonitor()
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTIVE MAINTENANCE - COMPLETE TRAINING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load Data
    logger.info("\n[Step 1/7] Loading Dataset...")
    t0 = time.time()

    downloader = DataDownloader()
    train_df, test_df, rul_df = downloader.load_data()

    monitor.log_timing("Data Loading", time.time() - t0)

    DataLogger.log_dataset_stats(train_df, "Training Set")
    DataLogger.log_dataset_stats(test_df, "Test Set")

    # Step 2: Preprocess Data
    logger.info("\n[Step 2/7] Preprocessing Data...")
    t0 = time.time()

    preprocessor = DataPreprocessor()
    sensor_cols = [col for col in train_df.columns if col.startswith("sensor_")]

    train_norm, norm_params = preprocessor.normalize_data(train_df, sensor_cols)
    test_norm, _ = preprocessor.normalize_data(test_df, sensor_cols)

    train_labeled = preprocessor.add_rul_labels(
        train_norm, rul_df, threshold_days=settings.RUL_THRESHOLD_DAYS
    )

    monitor.log_timing("Data Preprocessing", time.time() - t0)

    # Step 3: Anomaly Detection
    logger.info("\n[Step 3/7] Detecting Anomalies...")
    t0 = time.time()

    detector = AnomalyDetector(contamination=0.05)
    X_for_anomaly = train_labeled[sensor_cols].values
    anomalies, scores = detector.fit_detect(X_for_anomaly)

    train_labeled["is_anomaly"] = anomalies
    n_anomalies = (anomalies == -1).sum()
    logger.info(f"Found {n_anomalies} anomalies ({100*n_anomalies/len(anomalies):.1f}%)")

    monitor.log_timing("Anomaly Detection", time.time() - t0)

    # Step 4: Feature Engineering
    logger.info("\n[Step 4/7] Engineering Features...")
    t0 = time.time()

    engineer = FeatureEngineer()
    train_features = engineer.engineer_features(train_labeled)
    test_features = engineer.engineer_features(test_norm)

    monitor.log_timing("Feature Engineering", time.time() - t0)

    # Prepare model inputs
    feature_cols = [
        col
        for col in train_features.columns
        if not col.startswith("sensor_")
        and col
        not in [
            "machine_id",
            "cycle",
            "max_cycles",
            "cycles_to_failure",
            "failure_imminent",
            "is_anomaly",
        ]
    ]

    X_train = train_features[feature_cols].values
    y_train_clf = train_features["failure_imminent"].values
    y_train_reg = train_features["cycles_to_failure"].values

    X_test = test_features[feature_cols].values

    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(X_train)}")

    # Data Logger
    DataLogger.log_class_distribution(y_train_clf, ["Normal", "Failure Imminent"])

    # Step 5: Train Classification Models
    logger.info("\n[Step 5/7] Training Classification Models...")
    t0 = time.time()

    X_train_clf, X_val_clf, y_train_clf_split, y_val_clf = train_test_split(
        X_train,
        y_train_clf,
        test_size=settings.VALIDATION_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y_train_clf,
    )

    trainer = ModelTrainer()
    clf_results = trainer.train_classification_models(
        X_train_clf, X_val_clf, y_train_clf_split, y_val_clf, feature_names=feature_cols
    )

    monitor.log_timing("Classification Training", time.time() - t0)

    # Step 6: Train Regression Models
    logger.info("\n[Step 6/7] Training Regression Models...")
    t0 = time.time()

    X_train_reg, X_val_reg, y_train_reg_split, y_val_reg = train_test_split(
        X_train,
        y_train_reg,
        test_size=settings.VALIDATION_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    reg_results = trainer.train_regression_models(
        X_train_reg, X_val_reg, y_train_reg_split, y_val_reg
    )

    monitor.log_timing("Regression Training", time.time() - t0)

    # Step 7: Save Models and Artifacts
    logger.info("\n[Step 7/7] Saving Models and Artifacts...")
    t0 = time.time()

    manager = ModelManager()

    # Save best classifier
    best_clf_name = max(
        clf_results.items(),
        key=lambda x: x[1]["metrics"]["f1"],
    )[0]
    best_clf_model = clf_results[best_clf_name]["model"]
    manager.save_model(best_clf_model, "best_classifier")

    # Save best regressor
    best_reg_name = min(
        reg_results.items(),
        key=lambda x: x[1]["metrics"]["rmse"],
    )[0]
    best_reg_model = reg_results[best_reg_name]["model"]
    manager.save_model(best_reg_model, "best_regressor")

    # Save metadata
    metadata = {
        "feature_count": len(feature_cols),
        "train_samples": len(X_train),
        "sensor_count": len(sensor_cols),
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
    }
    manager.save_features_metadata(feature_cols, metadata)

    monitor.log_timing("Model Saving", time.time() - t0)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)

    logger.info(f"\n✅ Best Classification Model: {best_clf_name}")
    logger.info(f"   F1 Score: {clf_results[best_clf_name]['metrics']['f1']:.4f}")
    logger.info(f"   ROC-AUC: {clf_results[best_clf_name]['metrics']['roc_auc']:.4f}")

    logger.info(f"\n✅ Best Regression Model: {best_reg_name}")
    logger.info(f"   RMSE: {reg_results[best_reg_name]['metrics']['rmse']:.4f}")
    logger.info(f"   MAE: {reg_results[best_reg_name]['metrics']['mae']:.4f}")

    logger.info(f"\n📊 Features Engineered: {len(feature_cols)}")
    logger.info(f"🤖 Models Trained: {len(clf_results) + len(reg_results)}")

    monitor.summary()

    return {
        "clf_results": clf_results,
        "reg_results": reg_results,
        "features": feature_cols,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_complete_pipeline()

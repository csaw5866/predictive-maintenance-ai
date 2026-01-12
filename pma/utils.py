"""
Utility functions for the platform
"""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from pma.config import settings
from pma.logger import logger


class ModelManager:
    """Manage model loading, saving, and serving"""

    def __init__(self, models_dir: str = settings.MODELS_PATH):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, model_name: str) -> None:
        """Save model to disk"""
        path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, path)
        logger.info(f"Saved model: {path}")

    def load_model(self, model_name: str) -> Optional[Any]:
        """Load model from disk"""
        path = self.models_dir / f"{model_name}.pkl"

        if not path.exists():
            logger.warning(f"Model not found: {path}")
            return None

        model = joblib.load(path)
        logger.info(f"Loaded model: {path}")
        return model

    def save_features_metadata(self, feature_names: list[str], metadata: dict) -> None:
        """Save feature names and metadata"""
        data = {"feature_names": feature_names, "metadata": metadata}

        path = self.models_dir / "features.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved features metadata: {path}")

    def load_features_metadata(self) -> Optional[dict]:
        """Load feature names and metadata"""
        path = self.models_dir / "features.json"

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        return data


class MetricsReporter:
    """Generate comprehensive metrics reports"""

    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] = None) -> str:
        """Generate classification report"""
        if class_names is None:
            class_names = ["Normal", "Failure"]

        report = classification_report(y_true, y_pred, target_names=class_names)
        return report

    @staticmethod
    def confusion_matrix_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
        """Analyze confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
        }

    @staticmethod
    def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Analyze prediction residuals"""
        residuals = y_true - y_pred

        return {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
            "median_absolute_error": float(np.median(np.abs(residuals))),
        }


class DataLogger:
    """Log data statistics and quality metrics"""

    @staticmethod
    def log_dataset_stats(df: pd.DataFrame, name: str = "Dataset") -> None:
        """Log comprehensive dataset statistics"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {name}")
        logger.info(f"{'='*60}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"\nData types:\n{df.dtypes}")
        logger.info(f"\nMissing values:\n{df.isnull().sum()}")
        logger.info(f"\nBasic statistics:\n{df.describe()}")

    @staticmethod
    def log_feature_importance(importance: dict[str, float], top_n: int = 20) -> None:
        """Log feature importance scores"""
        logger.info(f"\nTop {top_n} Important Features:")
        logger.info("=" * 60)

        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for i, (feature, score) in enumerate(sorted_features[:top_n], 1):
            logger.info(f"{i:2d}. {feature:40s} {score:.6f}")

    @staticmethod
    def log_class_distribution(y: np.ndarray, class_names: list[str] = None) -> None:
        """Log class distribution for classification"""
        if class_names is None:
            class_names = ["Class " + str(i) for i in range(len(np.unique(y)))]

        logger.info("\nClass Distribution:")
        logger.info("=" * 60)

        unique, counts = np.unique(y, return_counts=True)

        for class_id, count in zip(unique, counts):
            pct = 100 * count / len(y)
            logger.info(f"{class_names[class_id]:20s}: {count:6d} ({pct:5.1f}%)")


class PerformanceMonitor:
    """Monitor and log runtime performance"""

    def __init__(self):
        self.metrics = {}

    def log_timing(self, task_name: str, duration_seconds: float) -> None:
        """Log task execution time"""
        self.metrics[task_name] = duration_seconds
        logger.info(f"⏱️  {task_name}: {duration_seconds:.2f}s")

    def summary(self) -> None:
        """Print performance summary"""
        if not self.metrics:
            return

        logger.info("\n" + "=" * 60)
        logger.info("Performance Summary")
        logger.info("=" * 60)

        total_time = sum(self.metrics.values())

        for task, duration in sorted(self.metrics.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * duration / total_time
            logger.info(f"{task:30s}: {duration:7.2f}s ({pct:5.1f}%)")

        logger.info(f"\n{'Total':30s}: {total_time:7.2f}s")

"""
Anomaly detection module for identifying unusual sensor patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pma.logger import logger


class AnomalyDetector:
    """Detect anomalies in sensor data"""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize detector

        Args:
            contamination: Proportion of outliers (0.0 to 0.5)
            random_state: For reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.logger = logger

    def fit(self, X: np.ndarray) -> None:
        """
        Fit anomaly detector on training data

        Args:
            X: Training feature matrix
        """
        X_scaled = self.scaler.fit_transform(X)
        self.detector.fit(X_scaled)
        self.logger.info("Anomaly detector fitted")

    def detect(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in data

        Args:
            X: Feature matrix to check

        Returns:
            Tuple of (predictions, scores)
            -1 = anomaly, 1 = normal
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.detector.predict(X_scaled)
        scores = self.detector.score_samples(X_scaled)

        n_anomalies = (predictions == -1).sum()
        self.logger.info(f"Detected {n_anomalies} anomalies out of {len(X)} samples")

        return predictions, scores

    def fit_detect(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit and detect in one step"""
        self.fit(X)
        return self.detect(X)


class SensorValidator:
    """Validate sensor readings for quality"""

    @staticmethod
    def check_sensor_range(df: pd.DataFrame, sensor_cols: list[str], bounds: dict[str, tuple] = None) -> pd.DataFrame:
        """
        Check if sensor values are within expected ranges

        Args:
            df: Input dataframe
            sensor_cols: List of sensor columns
            bounds: Dict of {sensor_name: (min, max)} bounds

        Returns:
            DataFrame with validation column
        """
        df = df.copy()
        valid = pd.Series(True, index=df.index)

        if bounds is None:
            bounds = {col: (df[col].quantile(0.01), df[col].quantile(0.99)) for col in sensor_cols}

        for col in sensor_cols:
            if col in bounds:
                lower, upper = bounds[col]
                valid = valid & (df[col] >= lower) & (df[col] <= upper)

        df["sensor_valid"] = valid.astype(int)
        return df

    @staticmethod
    def check_missing_values(df: pd.DataFrame, sensor_cols: list[str], max_missing_pct: float = 0.05) -> bool:
        """
        Check for excessive missing values

        Args:
            df: Input dataframe
            sensor_cols: List of sensor columns
            max_missing_pct: Maximum allowed missing percentage

        Returns:
            True if data is valid
        """
        for col in sensor_cols:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > max_missing_pct:
                logger.warning(f"{col}: {missing_pct*100:.1f}% missing values")
                return False

        return True

    @staticmethod
    def check_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, sensor_cols: list[str], threshold: float = 2.0) -> dict[str, bool]:
        """
        Detect data drift using distribution shift

        Args:
            reference_df: Reference/training data
            current_df: Current/test data
            sensor_cols: List of sensor columns
            threshold: Std dev threshold for drift

        Returns:
            Dict of {sensor: has_drift}
        """
        drift_results = {}

        for col in sensor_cols:
            ref_mean = reference_df[col].mean()
            ref_std = reference_df[col].std()

            curr_mean = current_df[col].mean()
            z_score = abs((curr_mean - ref_mean) / (ref_std + 1e-8))

            drift_results[col] = z_score > threshold

        n_drifted = sum(drift_results.values())
        logger.info(f"Data drift detected in {n_drifted}/{len(sensor_cols)} sensors")

        return drift_results

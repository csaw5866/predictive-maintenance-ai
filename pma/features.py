"""
Feature engineering module for time-series data
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

from pma.config import settings
from pma.logger import logger


class FeatureEngineer:
    """Engineering temporal and statistical features"""

    def __init__(self, rolling_window: int = settings.ROLLING_WINDOW_SIZE):
        self.rolling_window = rolling_window
        self.logger = logger

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive feature set for predictive modeling

        Args:
            df: Input dataframe with sensor data

        Returns:
            Dataframe with engineered features
        """
        df = df.copy()

        # Identify sensor columns
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
        self.logger.info(f"Engineering features from {len(sensor_cols)} sensors")

        # Group by machine for time-series features
        for machine_id in df["machine_id"].unique():
            mask = df["machine_id"] == machine_id
            machine_df = df[mask].copy().reset_index(drop=True)

            # Rolling statistics
            rolling_features = self._compute_rolling_stats(machine_df, sensor_cols)

            # Lag features
            lag_features = self._compute_lag_features(machine_df, sensor_cols)

            # FFT features
            fft_features = self._compute_fft_features(machine_df, sensor_cols)

            # Health indices
            health_features = self._compute_health_indices(machine_df, sensor_cols)

            # Combine all features
            idx = df[mask].index
            for feat_df in [rolling_features, lag_features, fft_features, health_features]:
                for col in feat_df.columns:
                    if col not in df.columns:
                        df.loc[idx, col] = feat_df[col].values

        # Fill NaN values
        df = df.fillna(df.mean(numeric_only=True))

        self.logger.info(f"Total features: {len(df.columns)}")
        return df

    def _compute_rolling_stats(
        self, df: pd.DataFrame, sensor_cols: list[str], window: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute rolling window statistics"""
        if window is None:
            window = self.rolling_window

        features = pd.DataFrame(index=df.index)

        for col in sensor_cols:
            features[f"{col}_rolling_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
            features[f"{col}_rolling_std_{window}"] = df[col].rolling(window, min_periods=1).std()
            features[f"{col}_rolling_min_{window}"] = df[col].rolling(window, min_periods=1).min()
            features[f"{col}_rolling_max_{window}"] = df[col].rolling(window, min_periods=1).max()

        return features

    def _compute_lag_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Compute lagged features"""
        features = pd.DataFrame(index=df.index)
        lags = settings.LAG_FEATURES

        for col in sensor_cols:
            for lag in lags:
                features[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return features

    def _compute_fft_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Compute FFT features for frequency domain analysis"""
        features = pd.DataFrame(index=df.index)

        for col in sensor_cols:
            if len(df) >= 2 * self.rolling_window:
                # Compute FFT
                values = df[col].values
                fft = np.abs(np.fft.fft(values))
                freqs = np.fft.fftfreq(len(values))

                # Top 5 frequency components
                top_freqs = np.argsort(fft)[-5:]
                for i, idx in enumerate(top_freqs):
                    features[f"{col}_fft_component_{i}"] = fft[idx]

                # Power spectrum
                features[f"{col}_fft_power"] = np.sum(fft ** 2)

        return features

    def _compute_health_indices(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Compute degradation and health indices"""
        features = pd.DataFrame(index=df.index)

        # Sensor correlation matrix
        sensor_corr = df[sensor_cols].corr()
        avg_corr = sensor_corr.values[np.triu_indices_from(sensor_corr.values, k=1)].mean()
        features["sensor_correlation_avg"] = avg_corr

        # Coefficient of variation per cycle
        for col in sensor_cols:
            features[f"{col}_cv"] = df[col].rolling(self.rolling_window, min_periods=1).apply(
                lambda x: x.std() / (x.mean() + 1e-8), raw=True
            )

        # Trend analysis
        for col in sensor_cols:
            features[f"{col}_trend"] = df[col].rolling(self.rolling_window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
            )

        # Overall system health score (inverse of normalized mean sensor values)
        all_sensors_mean = df[sensor_cols].mean(axis=1)
        features["system_health_score"] = 1.0 / (1.0 + (all_sensors_mean - all_sensors_mean.min()) / (all_sensors_mean.max() - all_sensors_mean.min() + 1e-8))

        return features

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Scale features using training data statistics"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.logger.info("Features scaled using StandardScaler")
        return X_train_scaled, X_test_scaled

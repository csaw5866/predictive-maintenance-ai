"""
Tests for feature engineering
"""

import pandas as pd
import pytest
from pma.features import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering"""

    def test_engineer_features(self, sample_data):
        """Test feature engineering pipeline"""
        engineer = FeatureEngineer(rolling_window=10)
        features = engineer.engineer_features(sample_data)

        # Check output
        assert len(features) == len(sample_data)
        assert len(features.columns) > len(sample_data.columns)

    def test_rolling_stats(self, sample_data):
        """Test rolling statistics"""
        engineer = FeatureEngineer(rolling_window=5)

        sensor_cols = [col for col in sample_data.columns if col.startswith("sensor_")]
        features = engineer._compute_rolling_stats(sample_data, sensor_cols, window=5)

        # Check features were created
        expected_features = len(sensor_cols) * 4  # mean, std, min, max
        assert len(features.columns) == expected_features

    def test_lag_features(self, sample_data):
        """Test lag feature creation"""
        engineer = FeatureEngineer()

        sensor_cols = [col for col in sample_data.columns if col.startswith("sensor_")]
        features = engineer._compute_lag_features(sample_data, sensor_cols)

        # Check features were created
        assert len(features.columns) > 0

    def test_fft_features(self, sample_data):
        """Test FFT feature creation"""
        engineer = FeatureEngineer()

        sensor_cols = [col for col in sample_data.columns if col.startswith("sensor_")]
        features = engineer._compute_fft_features(sample_data, sensor_cols)

        # Check features were created
        assert len(features.columns) > 0

    def test_health_indices(self, sample_data):
        """Test health index creation"""
        engineer = FeatureEngineer()

        sensor_cols = [col for col in sample_data.columns if col.startswith("sensor_")]
        features = engineer._compute_health_indices(sample_data, sensor_cols)

        # Check features were created
        assert len(features.columns) > 0
        assert "system_health_score" in features.columns

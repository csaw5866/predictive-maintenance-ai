"""
Tests for data loading and preprocessing
"""

import pandas as pd
import pytest
from pma.data import DataDownloader, DataPreprocessor


class TestDataDownloader:
    """Test data downloading and loading"""

    def test_synthetic_data_creation(self):
        """Test synthetic data generation"""
        downloader = DataDownloader()
        downloader._create_synthetic_nasa_data()

    def test_data_loading(self):
        """Test loading data"""
        downloader = DataDownloader()
        train_df, test_df, rul_df = downloader.load_data()

        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(rul_df) > 0

        assert "machine_id" in train_df.columns
        assert "cycle" in train_df.columns


class TestDataPreprocessor:
    """Test data preprocessing"""

    def test_normalization(self, sample_data):
        """Test data normalization"""
        preprocessor = DataPreprocessor()

        sensor_cols = [col for col in sample_data.columns if col.startswith("sensor_")]
        normalized, params = preprocessor.normalize_data(sample_data, sensor_cols)

        assert len(normalized) == len(sample_data)
        assert len(params) == len(sensor_cols)

        # Check normalization applied
        for col in sensor_cols:
            assert abs(normalized[col].mean()) < 0.1  # Should be ~0
            assert abs(normalized[col].std() - 1.0) < 0.1  # Should be ~1

    def test_rul_labeling(self, sample_data, rul_data):
        """Test RUL label creation"""
        preprocessor = DataPreprocessor()

        # Create max_cycles for testing
        sample_data["max_cycles"] = 200

        labeled = preprocessor.add_rul_labels(sample_data, rul_data, threshold_days=30)

        assert "failure_imminent" in labeled.columns
        assert "cycles_to_failure" in labeled.columns
        assert labeled["failure_imminent"].dtype == int

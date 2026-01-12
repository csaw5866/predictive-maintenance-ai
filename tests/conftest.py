"""
Test configuration and fixtures
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample sensor data for testing"""
    n_samples = 100
    n_sensors = 21

    data = {
        "machine_id": np.repeat(range(1, 5), n_samples // 4),
        "cycle": np.tile(range(1, n_samples // 4 + 1), 4),
    }

    # Add operational settings
    data["op_setting_1"] = np.random.uniform(0, 100, n_samples)
    data["op_setting_2"] = np.random.uniform(0, 100, n_samples)
    data["op_setting_3"] = np.random.uniform(0, 100, n_samples)

    # Add sensor readings
    for i in range(1, n_sensors + 1):
        data[f"sensor_{i}"] = np.random.normal(100, 10, n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def rul_data():
    """Create RUL labels for testing"""
    return pd.DataFrame({"rul": [100, 150, 120, 140]})


@pytest.fixture
def feature_data(sample_data):
    """Create feature-engineered data"""
    features = sample_data.copy()

    # Add some engineered features
    sensor_cols = [col for col in features.columns if col.startswith("sensor_")]
    for col in sensor_cols:
        features[f"{col}_rolling_mean"] = features[col].rolling(5).mean()
        features[f"{col}_lag_1"] = features[col].shift(1)

    features = features.fillna(features.mean(numeric_only=True))
    return features

"""
Data loading and preprocessing module
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from pma.config import settings
from pma.logger import logger


class DataDownloader:
    """Download and manage industrial datasets"""

    def __init__(self, data_dir: str = settings.DATASET_PATH):
        """Initialize downloader"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_nasa_cmapss(self) -> dict[str, pd.DataFrame]:
        """
        Download NASA C-MAPSS Turbofan Engine Degradation Dataset
        
        Returns:
            Dictionary with train and test dataframes
        """
        logger.info("Checking for NASA C-MAPSS dataset...")

        train_file = self.data_dir / "train_FD001.txt"
        test_file = self.data_dir / "test_FD001.txt"
        rul_file = self.data_dir / "RUL_FD001.txt"

        # If files exist, load them
        if train_file.exists() and test_file.exists():
            logger.info("Dataset files found locally, loading...")
            return self._load_nasa_cmapss_local()

        # Otherwise create synthetic data for demo
        logger.warning("NASA dataset not found locally. Creating synthetic dataset for demo...")
        return self._create_synthetic_nasa_data()

    def _load_nasa_cmapss_local(self) -> dict[str, pd.DataFrame]:
        """Load existing NASA C-MAPSS data"""
        train_file = self.data_dir / "train_FD001.txt"
        test_file = self.data_dir / "test_FD001.txt"
        rul_file = self.data_dir / "RUL_FD001.txt"

        # Column names for NASA dataset
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        cols = ["machine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + sensor_cols
        col_names = cols

        train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)
        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=col_names)
        rul_df = pd.read_csv(rul_file, header=None, names=["rul"])

        return {"train": train_df, "test": test_df, "rul": rul_df}

    def _create_synthetic_nasa_data(self) -> dict[str, pd.DataFrame]:
        """Create synthetic turbofan dataset"""
        logger.info("Creating synthetic turbofan engine dataset...")

        n_machines = 100
        n_cycles_per_machine = 300
        n_sensors = 21

        data = []
        rul_data = []

        for machine_id in range(1, n_machines + 1):
            n_cycles = np.random.randint(150, n_cycles_per_machine)
            rul_at_end = np.random.randint(1, 50)

            for cycle in range(1, n_cycles + 1):
                # Operational settings
                op_setting_1 = np.random.uniform(0, 100)
                op_setting_2 = np.random.uniform(0, 100)
                op_setting_3 = np.random.uniform(0, 100)

                # Degradation factor increases over time
                degradation = (cycle / n_cycles) * 50

                # Sensor values with degradation
                sensors = []
                for i in range(n_sensors):
                    base_value = 100 + np.random.normal(0, 2)
                    degraded_value = base_value - degradation + np.random.normal(0, 1)
                    sensors.append(degraded_value)

                row = [machine_id, cycle, op_setting_1, op_setting_2, op_setting_3] + sensors
                data.append(row)

            rul_data.append(rul_at_end)

        # Create dataframes
        sensor_cols = [f"sensor_{i}" for i in range(1, n_sensors + 1)]
        col_names = ["machine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + sensor_cols

        train_df = pd.DataFrame(data[: len(data) // 2], columns=col_names)
        test_df = pd.DataFrame(data[len(data) // 2 :], columns=col_names)
        rul_df = pd.DataFrame(rul_data, columns=["rul"])

        logger.info(f"Created synthetic data: {len(train_df)} train, {len(test_df)} test samples")

        return {"train": train_df, "test": test_df, "rul": rul_df}

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and return train, test, and RUL data"""
        data = self.download_nasa_cmapss()
        return data["train"], data["test"], data["rul"]


class DataPreprocessor:
    """Preprocess time-series data"""

    def __init__(self):
        self.logger = logger

    def normalize_data(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Normalize sensor data using z-score normalization

        Args:
            df: Input dataframe
            columns: Columns to normalize (None = all numeric)

        Returns:
            Normalized dataframe and normalization parameters
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        norm_params = {}
        df_normalized = df.copy()

        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            norm_params[col] = {"mean": mean, "std": std}
            df_normalized[col] = (df[col] - mean) / (std + 1e-8)

        self.logger.info(f"Normalized {len(columns)} columns")
        return df_normalized, norm_params

    def add_rul_labels(
        self, df: pd.DataFrame, rul_values: pd.DataFrame, threshold_days: int = 30
    ) -> pd.DataFrame:
        """
        Add RUL (Remaining Useful Life) and failure labels

        Args:
            df: Input dataframe with machine_id and cycle columns
            rul_values: Series with RUL values per machine
            threshold_days: Days until failure for classification label

        Returns:
            Dataframe with RUL and failure_imminent columns
        """
        df = df.copy()

        # Add RUL for each machine
        def get_max_cycle(machine_id):
            max_cycles = df[df["machine_id"] == machine_id]["cycle"].max()
            return max_cycles

        df["max_cycles"] = df["machine_id"].apply(get_max_cycle)
        df["cycles_to_failure"] = df["max_cycles"] - df["cycle"]

        # Classification label: failure within threshold
        df["failure_imminent"] = (df["cycles_to_failure"] <= threshold_days).astype(int)

        self.logger.info(
            f"Added RUL labels. {(df['failure_imminent'] == 1).sum()} failure samples"
        )

        return df

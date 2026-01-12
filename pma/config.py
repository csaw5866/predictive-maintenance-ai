"""
Configuration management for the predictive maintenance platform
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # Project
    PROJECT_NAME: str = "Predictive Maintenance AI"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/predictive_maintenance"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "predictive-maintenance"
    MLFLOW_BACKEND_STORE_URI: str = "./mlruns"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Dashboard
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 8501

    # Data paths
    DATASET_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"
    MODELS_PATH: str = "./models"

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # Feature engineering
    RUL_THRESHOLD_DAYS: int = 30  # Days until failure for classification
    ROLLING_WINDOW_SIZE: int = 50  # Time steps for rolling features
    LAG_FEATURES: list[int] = [1, 5, 10, 20]

    # Model training
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    N_JOBS: int = -1

    # Dataset specific
    NASA_DATASET_URL: str = "https://ti.arc.nasa.gov/c-mapss/"
    USE_SAMPLE_DATA: bool = False  # If True, creates synthetic data

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()

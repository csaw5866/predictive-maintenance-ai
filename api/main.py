"""
FastAPI main application
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pma.config import settings
from pma.logger import logger
from pma.schemas import FailurePrediction, MachineHealthResponse, PredictionRequest, RULPrediction

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Predictive Maintenance AI API",
    version=settings.PROJECT_VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
_models_cache = {}


def load_models() -> dict:
    """Load trained models from disk"""
    global _models_cache

    if _models_cache:
        return _models_cache

    models_dir = Path(settings.MODELS_PATH)

    if not models_dir.exists():
        logger.warning("Models directory not found, creating dummy models...")
        return {}

    try:
        # Load classification model
        clf_path = models_dir / "best_classifier.pkl"
        if clf_path.exists():
            _models_cache["classifier"] = joblib.load(clf_path)
            logger.info("Loaded classifier model")

        # Load regression model
        reg_path = models_dir / "best_regressor.pkl"
        if reg_path.exists():
            _models_cache["regressor"] = joblib.load(reg_path)
            logger.info("Loaded regressor model")

    except Exception as e:
        logger.error(f"Error loading models: {e}")

    return _models_cache


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Predictive Maintenance API...")
    load_models()
    logger.info("API startup complete")


@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "predictive-maintenance-api",
    }


@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Get API metrics"""
    return {
        "models_loaded": len(_models_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict/failure", response_model=list[FailurePrediction], tags=["predictions"])
async def predict_failure(request: PredictionRequest) -> list[FailurePrediction]:
    """
    Predict machine failure probability

    Args:
        request: Sensor readings

    Returns:
        Failure predictions with probabilities
    """
    models = load_models()

    if "classifier" not in models:
        raise HTTPException(
            status_code=503, detail="Failure prediction model not available"
        )

    predictions = []

    try:
        for reading in request.readings:
            # Prepare features (simplified - in production, would use full feature engineering)
            sensor_values = list(reading.sensors.values())
            features = np.array(
                [[reading.op_setting_1, reading.op_setting_2, reading.op_setting_3]]
                + [[v] for v in sensor_values]
            ).flatten().reshape(1, -1)

            # Predict
            clf = models["classifier"]
            pred_proba = clf.predict_proba(features)[0, 1]
            pred_label = clf.predict(features)[0]

            predictions.append(
                FailurePrediction(
                    machine_id=reading.machine_id,
                    cycle=reading.cycle,
                    failure_probability=float(pred_proba),
                    failure_imminent=bool(pred_label),
                    confidence=float(max(clf.predict_proba(features)[0])),
                    model_name="xgboost",
                )
            )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return predictions


@app.post("/predict/rul", response_model=list[RULPrediction], tags=["predictions"])
async def predict_rul(request: PredictionRequest) -> list[RULPrediction]:
    """
    Predict Remaining Useful Life

    Args:
        request: Sensor readings

    Returns:
        RUL predictions
    """
    models = load_models()

    if "regressor" not in models:
        raise HTTPException(status_code=503, detail="RUL prediction model not available")

    predictions = []

    try:
        for reading in request.readings:
            # Prepare features
            sensor_values = list(reading.sensors.values())
            features = np.array(
                [[reading.op_setting_1, reading.op_setting_2, reading.op_setting_3]]
                + [[v] for v in sensor_values]
            ).flatten().reshape(1, -1)

            # Predict
            regressor = models["regressor"]
            rul_pred = regressor.predict(features)[0]

            predictions.append(
                RULPrediction(
                    machine_id=reading.machine_id,
                    cycle=reading.cycle,
                    estimated_rul_cycles=max(0, int(rul_pred)),
                    confidence=0.85,  # Placeholder
                    model_name="lightgbm",
                )
            )

    except Exception as e:
        logger.error(f"RUL prediction error: {e}")
        raise HTTPException(status_code=500, detail="RUL prediction failed")

    return predictions


@app.get("/machines/{machine_id}/health", response_model=MachineHealthResponse, tags=["monitoring"])
async def get_machine_health(machine_id: int):
    """Get machine health status"""
    # Placeholder - would fetch from database in production
    return MachineHealthResponse(
        machine_id=machine_id,
        current_cycle=100,
        health_score=0.75,
        failure_probability=0.15,
        estimated_rul=150,
        status="degrading",
        last_update=datetime.utcnow().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
    )

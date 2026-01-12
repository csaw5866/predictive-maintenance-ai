"""
FastAPI inference server for predictive maintenance
"""

from typing import Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Single sensor reading"""

    machine_id: int = Field(..., description="Unique machine identifier")
    cycle: int = Field(..., description="Operating cycle number")
    op_setting_1: float = Field(..., description="Operating setting 1")
    op_setting_2: float = Field(..., description="Operating setting 2")
    op_setting_3: float = Field(..., description="Operating setting 3")
    sensors: dict[str, float] = Field(..., description="Sensor readings")


class PredictionRequest(BaseModel):
    """Prediction request"""

    readings: list[SensorReading] = Field(..., description="List of sensor readings")


class FailurePrediction(BaseModel):
    """Failure prediction result"""

    machine_id: int
    cycle: int
    failure_probability: float = Field(..., ge=0, le=1)
    failure_imminent: bool
    confidence: float
    model_name: str


class RULPrediction(BaseModel):
    """Remaining Useful Life prediction"""

    machine_id: int
    cycle: int
    estimated_rul_cycles: int
    estimated_rul_days: Optional[float] = None
    confidence: float
    model_name: str


class MachineHealthResponse(BaseModel):
    """Machine health status"""

    machine_id: int
    current_cycle: int
    health_score: float
    failure_probability: float
    estimated_rul: int
    status: str  # "healthy", "degrading", "critical"
    last_update: str

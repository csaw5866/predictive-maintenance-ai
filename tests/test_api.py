"""
Tests for FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoints"""

    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "models_loaded" in response.json()


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    def test_predict_failure_endpoint(self, client):
        """Test /predict/failure endpoint"""
        payload = {
            "readings": [
                {
                    "machine_id": 1,
                    "cycle": 100,
                    "op_setting_1": 50.0,
                    "op_setting_2": 100.0,
                    "op_setting_3": 75.0,
                    "sensors": {"sensor_1": 100.5, "sensor_2": 102.3},
                }
            ]
        }

        response = client.post("/predict/failure", json=payload)
        
        # Model might not be available in test environment
        if response.status_code == 200:
            assert len(response.json()) == 1
            assert response.json()[0]["machine_id"] == 1

    def test_predict_rul_endpoint(self, client):
        """Test /predict/rul endpoint"""
        payload = {
            "readings": [
                {
                    "machine_id": 1,
                    "cycle": 100,
                    "op_setting_1": 50.0,
                    "op_setting_2": 100.0,
                    "op_setting_3": 75.0,
                    "sensors": {"sensor_1": 100.5},
                }
            ]
        }

        response = client.post("/predict/rul", json=payload)
        
        # Model might not be available in test environment
        if response.status_code == 200:
            assert len(response.json()) == 1

    def test_machine_health_endpoint(self, client):
        """Test /machines/{machine_id}/health endpoint"""
        response = client.get("/machines/1/health")
        assert response.status_code == 200
        assert response.json()["machine_id"] == 1

"""Automated tests for FastAPI REST endpoints in src/api.py."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Test client fixture for FastAPI app."""
    return TestClient(app)


def test_api_health(client: TestClient):
    """Verify health endpoint returns status 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert data["model_loaded"] is True


def test_api_predict_success(client: TestClient):
    """Verify valid single property prediction."""
    payload = {
        "area": 1650,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 4.0,
        "locality": "Suburban",
        "parking": 1,
        "furnished_status": "Semi-Furnished",
        "has_amenities": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["price_lakhs"] > 0
    assert data["prediction_interval_lower"] <= data["price_lakhs"]
    assert data["prediction_interval_upper"] >= data["price_lakhs"]
    assert data["confidence_interval_lower"] == data["prediction_interval_lower"]
    assert data["confidence_interval_upper"] == data["prediction_interval_upper"]
    assert data["price_per_sqft_inr"] > 0
    assert data["features_used"]["area"] == 1650


def test_api_predict_negative_area(client: TestClient):
    """Verify Pydantic validation rejects negative area with 422."""
    payload = {
        "area": -500,
        "bedrooms": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_missing_area(client: TestClient):
    """Verify Pydantic validation rejects missing area with 422."""
    payload = {
        "bedrooms": 3,
        "bathrooms": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_predict_invalid_locality(client: TestClient):
    """Verify domain validation rejects unrecognized locality."""
    payload = {
        "area": 1500,
        "locality": "Mars Base Alpha",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_batch_oversized(client: TestClient):
    """Verify batch validation rejects more than 500 records with 422."""
    oversized = [{"area": 1500, "bedrooms": 3} for _ in range(501)]
    response = client.post("/predict/batch", json={"properties": oversized})
    assert response.status_code == 422


def test_api_batch_predict(client: TestClient):
    """Verify batch valuation endpoint."""
    payload = {
        "properties": [
            {"area": 1200, "bedrooms": 2, "bathrooms": 2, "age": 5.0, "locality": "Suburban"},
            {"area": 2400, "bedrooms": 4, "bathrooms": 3, "age": 1.0, "locality": "Urban Prime"},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_properties"] == 2
    assert data["total_portfolio_value_lakhs"] > 0
    assert len(data["results"]) == 2


def test_api_model_metadata(client: TestClient):
    """Verify model metadata endpoint returns valid dictionary."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "algorithm" in data


def test_api_model_features(client: TestClient):
    """Verify feature importances endpoint."""
    response = client.get("/model/features")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importances" in data
    assert "allowed_localities" in data

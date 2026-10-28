"""Integration tests for application components and model serving."""

import pytest
from pathlib import Path

from src.models import load_model_pipeline
from src.predictor import HousePricePredictor
from src.config import CHAMPION_MODEL_PATH, LEGACY_MODEL_PATH


def test_production_model_available():
    """Verify that a trained champion model exists and can be loaded."""
    assert CHAMPION_MODEL_PATH.exists() or LEGACY_MODEL_PATH.exists()
    pipeline = load_model_pipeline()
    assert pipeline is not None


def test_end_to_end_prediction():
    """Verify end-to-end prediction from user input to final valuation."""
    predictor = HousePricePredictor()
    user_inputs = {
        "area": 1850,
        "bedrooms": 3,
        "bathrooms": 3,
        "age": 4.5,
        "locality": "Urban Prime",
        "parking": 2,
        "furnished_status": "Furnished",
        "has_amenities": 1,
    }

    result = predictor.predict(user_inputs)
    assert result.price_lakhs > 50.0  # Prime 3BHK should be a substantial valuation
    assert result.confidence_interval_lower < result.price_lakhs < result.confidence_interval_upper
    assert result.price_per_sqft_inr > 3000.0

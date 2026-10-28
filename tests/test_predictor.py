"""Unit tests for src/predictor.py."""

import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import load_dataset
from src.preprocessing import build_pipeline
from src.predictor import (
    HousePricePredictor,
    PredictionResult,
    PredictionValidationError,
)
from src.config import ALL_FEATURES, MIN_PRICE_FLOOR_LAKHS


@pytest.fixture
def trained_predictor():
    """Fixture providing a ready HousePricePredictor backed by a fast LinearRegression pipeline."""
    df = load_dataset().head(100)
    X = df[ALL_FEATURES]
    y = df["price"]

    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)
    return HousePricePredictor(pipeline=pipeline)


def test_predictor_normal_prediction(trained_predictor: HousePricePredictor):
    """Verify normal inference execution and result schema."""
    input_sample = {
        "area": 1600,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 5,
        "locality": "Suburban",
        "parking": 1,
        "furnished_status": "Semi-Furnished",
        "has_amenities": 1,
    }

    result = trained_predictor.predict(input_sample)
    assert isinstance(result, PredictionResult)
    assert result.price_lakhs > 0
    assert result.prediction_interval_lower <= result.price_lakhs
    assert result.prediction_interval_upper >= result.price_lakhs
    assert result.confidence_interval_lower == result.prediction_interval_lower
    assert result.confidence_interval_upper == result.prediction_interval_upper
    assert result.price_per_sqft_inr > 0
    assert not result.clamped


def test_predictor_legacy_features_only(trained_predictor: HousePricePredictor):
    """Verify backwards-compatibility with original 4-feature inputs."""
    legacy_input = {
        "area": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 10,
    }

    result = trained_predictor.predict(legacy_input)
    assert isinstance(result, PredictionResult)
    assert result.price_lakhs > 0
    assert "locality" in result.features_used
    assert result.features_used["locality"] == "Suburban"


def test_predictor_floor_clamping(trained_predictor: HousePricePredictor):
    """Verify that predictions can never fall below the physical domain floor."""
    from sklearn.base import BaseEstimator, RegressorMixin

    class DummyNegativeRegressor(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def predict(self, X):
            return [-150.0] * len(X)

    from src.preprocessing import create_preprocessor
    from sklearn.pipeline import Pipeline
    dummy_pipe = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", DummyNegativeRegressor()),
    ])
    df = load_dataset().head(10)
    dummy_pipe.fit(df[ALL_FEATURES], df["price"])

    pred = HousePricePredictor(pipeline=dummy_pipe)
    res = pred.predict({"area": 1000, "bedrooms": 2, "bathrooms": 1, "age": 20})

    assert res.price_lakhs == MIN_PRICE_FLOOR_LAKHS
    assert res.clamped is True


def test_predictor_rejects_negative_area(trained_predictor: HousePricePredictor):
    """Verify that invalid negative area raises PredictionValidationError."""
    with pytest.raises(PredictionValidationError, match="outside valid domain"):
        trained_predictor.predict({"area": -500, "bedrooms": 2, "bathrooms": 1, "age": 5})


def test_predictor_rejects_invalid_locality(trained_predictor: HousePricePredictor):
    """Verify rejection of unsupported categorical option."""
    with pytest.raises(PredictionValidationError, match="Invalid value"):
        trained_predictor.predict({
            "area": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 5,
            "locality": "Outer Space",
        })


def test_predictor_batch_prediction(trained_predictor: HousePricePredictor):
    """Verify batch prediction on DataFrame."""
    df_batch = pd.DataFrame([
        {"area": 1200, "bedrooms": 2, "bathrooms": 2, "age": 5},
        {"area": 2200, "bedrooms": 4, "bathrooms": 3, "age": 2},
    ])

    batch_out = trained_predictor.predict_batch(df_batch)
    assert "predicted_price_lakhs" in batch_out.columns
    assert "prediction_interval_lower_lakhs" in batch_out.columns
    assert "prediction_interval_upper_lakhs" in batch_out.columns
    assert "price_per_sqft_inr" in batch_out.columns
    assert len(batch_out) == 2
    assert (batch_out["predicted_price_lakhs"] > 0).all()


def test_predictor_feature_importances_sum_100(trained_predictor: HousePricePredictor):
    """Verify that feature importances sum to exactly 100.0%."""
    importances = trained_predictor.get_feature_importances()
    assert len(importances) > 0
    assert round(sum(importances.values()), 1) == 100.0

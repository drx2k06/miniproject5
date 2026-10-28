"""
Safe inference engine and domain rule guardrails for house price prediction.
Provides input validation, physical domain clamping, confidence intervals,
and backwards compatibility for legacy 4-feature inputs.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    CHAMPION_MODEL_PATH,
    MODEL_METADATA_PATH,
    ALL_FEATURES,
    LEGACY_FEATURES,
    FEATURE_CONSTRAINTS,
    MIN_PRICE_FLOOR_LAKHS,
    DEFAULT_CONFIDENCE_PERCENTILE,
)
from src.models import load_model_pipeline


@dataclass
class PredictionResult:
    """
    Structured valuation output containing point estimate and uncertainty bounds.
    Note: prediction_interval_* represents the uncertainty bound for an individual
    observation (distinct from a confidence interval for the population mean).
    """
    price_lakhs: float
    confidence_interval_lower: float  # Backwards-compatible alias for prediction_interval_lower
    confidence_interval_upper: float  # Backwards-compatible alias for prediction_interval_upper
    price_per_sqft_inr: float
    clamped: bool
    features_used: Dict[str, Any]
    prediction_interval_lower: float = 0.0
    prediction_interval_upper: float = 0.0

    def __post_init__(self):
        if self.prediction_interval_lower == 0.0:
            self.prediction_interval_lower = self.confidence_interval_lower
        if self.prediction_interval_upper == 0.0:
            self.prediction_interval_upper = self.confidence_interval_upper


class PredictionValidationError(ValueError):
    """Raised when input features fail domain boundaries or validation rules."""
    pass


class HousePricePredictor:
    """
    Production-grade predictor service with domain safeguards and uncertainty bounds.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        pipeline: Optional[Pipeline] = None,
    ):
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = load_model_pipeline(model_path)

        self.metadata = self._load_metadata(metadata_path)
        self.residual_std = self._extract_residual_std()

    def _load_metadata(self, metadata_path: Optional[Path] = None) -> Dict[str, Any]:
        meta_p = Path(metadata_path) if metadata_path else MODEL_METADATA_PATH
        if meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _extract_residual_std(self) -> float:
        """Extract residual standard deviation for confidence interval calculation."""
        if "residual_stats" in self.metadata:
            return float(self.metadata["residual_stats"].get("residual_std", 5.0))
        if "test_metrics" in self.metadata:
            return float(self.metadata["test_metrics"].get("RMSE", 5.0))
        return 5.0  # Safe default fallback

    def get_feature_importances(self) -> Dict[str, float]:
        """Return cached or dynamically extracted feature importances."""
        if "feature_importances" in self.metadata and self.metadata["feature_importances"]:
            return self.metadata["feature_importances"]
        from src.evaluation import extract_feature_importances
        return extract_feature_importances(self.pipeline)

    def validate_and_normalize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate single-sample dictionary and fill default values for missing enhanced fields.
        """
        normalized: Dict[str, Any] = {}

        # 1. Check legacy/required fields
        for field in LEGACY_FEATURES:
            if field not in data:
                raise PredictionValidationError(f"Missing mandatory feature: '{field}'")
            val = data[field]
            try:
                val = float(val)
            except (ValueError, TypeError) as exc:
                raise PredictionValidationError(f"Feature '{field}' must be numeric, got {val}") from exc

            constraint = FEATURE_CONSTRAINTS[field]
            if val < constraint["min"] or val > constraint["max"]:
                raise PredictionValidationError(
                    f"Feature '{field}' value {val} is outside valid domain [{constraint['min']}, {constraint['max']}]"
                )
            normalized[field] = int(val) if constraint["type"] is int else val

        # 2. Check and fill enhanced features
        for field, constraint in FEATURE_CONSTRAINTS.items():
            if field in LEGACY_FEATURES:
                continue

            if field in data and data[field] is not None:
                val = data[field]
                if "allowed" in constraint:
                    if val not in constraint["allowed"]:
                        raise PredictionValidationError(
                            f"Invalid value '{val}' for '{field}'. Allowed: {constraint['allowed']}"
                        )
                    normalized[field] = val
                elif "min" in constraint:
                    val = float(val)
                    if val < constraint["min"] or val > constraint["max"]:
                        raise PredictionValidationError(
                            f"Feature '{field}' value {val} is outside valid domain [{constraint['min']}, {constraint['max']}]"
                        )
                    normalized[field] = int(val) if constraint["type"] is int else val
            else:
                # Fill default value for backwards compatibility
                normalized[field] = constraint["default"]

        return normalized

    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> PredictionResult:
        """
        Predict house valuation with domain clamping and uncertainty bounds.

        Args:
            input_data: Dictionary or single-row DataFrame.

        Returns:
            PredictionResult: Valued output with CI and telemetry.
        """
        if isinstance(input_data, pd.DataFrame):
            if len(input_data) != 1:
                raise ValueError("Single-row DataFrame expected for predict(). Use predict_batch() for multiple.")
            input_dict = input_data.iloc[0].to_dict()
        else:
            input_dict = dict(input_data)

        normalized = self.validate_and_normalize_input(input_dict)
        sample_df = pd.DataFrame([normalized])

        # Execute raw model inference
        raw_pred = float(self.pipeline.predict(sample_df)[0])

        # Enforce physical floor clamping: house prices cannot be negative or absurdly small
        clamped = False
        final_price = raw_pred
        if final_price < MIN_PRICE_FLOOR_LAKHS:
            final_price = MIN_PRICE_FLOOR_LAKHS
            clamped = True

        # Calculate 95% Prediction Interval (uncertainty bound for an individual property observation)
        z_score = 1.96  # Empirical Gaussian approximation
        margin = z_score * self.residual_std
        pi_lower = max(MIN_PRICE_FLOOR_LAKHS, round(final_price - margin, 2))
        pi_upper = round(final_price + margin, 2)

        # Price per square foot in INR: (Price in Lakhs * 100,000) / Area sqft
        area_sqft = normalized["area"]
        rate_per_sqft = round((final_price * 100_000.0) / area_sqft, 2)

        return PredictionResult(
            price_lakhs=round(final_price, 2),
            confidence_interval_lower=pi_lower,
            confidence_interval_upper=pi_upper,
            prediction_interval_lower=pi_lower,
            prediction_interval_upper=pi_upper,
            price_per_sqft_inr=rate_per_sqft,
            clamped=clamped,
            features_used=normalized,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for a collection of property records.
        """
        out_df = df.copy()
        prices = []
        pi_lowers = []
        pi_uppers = []
        rates = []

        for _, row in df.iterrows():
            res = self.predict(row.to_dict())
            prices.append(res.price_lakhs)
            pi_lowers.append(res.prediction_interval_lower)
            pi_uppers.append(res.prediction_interval_upper)
            rates.append(res.price_per_sqft_inr)

        out_df["predicted_price_lakhs"] = prices
        out_df["ci_lower_lakhs"] = pi_lowers
        out_df["ci_upper_lakhs"] = pi_uppers
        out_df["prediction_interval_lower_lakhs"] = pi_lowers
        out_df["prediction_interval_upper_lakhs"] = pi_uppers
        out_df["price_per_sqft_inr"] = rates

        return out_df

"""
FastAPI REST API Service for House Price Valuation Engine.
Provides high-performance typed endpoints for single property prediction,
batch portfolio scoring, model telemetry, and health checks.
"""

from typing import List, Dict, Any, Optional

# Compatibility patch for Starlette 1.x with FastAPI <= 0.115.6
import starlette.routing
_orig_router_init = starlette.routing.Router.__init__
def _patched_router_init(self, *args, **kwargs):
    kwargs.pop("on_startup", None)
    kwargs.pop("on_shutdown", None)
    return _orig_router_init(self, *args, **kwargs)
starlette.routing.Router.__init__ = _patched_router_init

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Ensure compatibility with Starlette 1.x
FastAPI.max_body_size = None

from src.predictor import HousePricePredictor, PredictionValidationError
from src.config import LOCALITIES, FURNISHED_STATUSES, CHAMPION_MODEL_PATH

# FastAPI App initialization
app = FastAPI(
    title="House Price Valuation Engine API",
    version="2.0.0",
    description="High-performance machine learning microservice for residential real estate valuation.",
)

# Global predictor instance
_predictor: Optional[HousePricePredictor] = None


def get_predictor_instance() -> HousePricePredictor:
    """Retrieve or initialize cached singleton predictor instance."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = HousePricePredictor()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifacts not available. Please train the model: {exc}",
            ) from exc
    return _predictor


# Pydantic Schemas
class PropertyRequest(BaseModel):
    area: float = Field(..., ge=100, le=15000, description="Property area in square feet", examples=[1650])
    bedrooms: int = Field(default=3, ge=1, le=12, description="Number of bedrooms (BHK)", examples=[3])
    bathrooms: int = Field(default=2, ge=1, le=10, description="Number of bathrooms", examples=[2])
    age: float = Field(default=5.0, ge=0.0, le=100.0, description="Age of property in years", examples=[4.0])
    locality: str = Field(
        default="Suburban",
        description=f"Neighborhood classification: {LOCALITIES}",
        examples=["Suburban"],
    )
    parking: int = Field(default=1, ge=0, le=10, description="Designated parking spaces", examples=[1])
    furnished_status: str = Field(
        default="Semi-Furnished",
        description=f"Furnishing level: {FURNISHED_STATUSES}",
        examples=["Semi-Furnished"],
    )
    has_amenities: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 if society amenities (gym, pool, security) are available, 0 otherwise",
        examples=[1],
    )


class ValuationResponse(BaseModel):
    price_lakhs: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence_interval_lower: float  # Backwards-compatible alias
    confidence_interval_upper: float  # Backwards-compatible alias
    price_per_sqft_inr: float
    clamped: bool
    features_used: Dict[str, Any]


class BatchValuationRequest(BaseModel):
    properties: List[PropertyRequest] = Field(..., min_length=1, max_length=500)


class BatchValuationResponse(BaseModel):
    total_properties: int
    total_portfolio_value_lakhs: float
    results: List[ValuationResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verify service readiness and model artifact loading."""
    try:
        pred = get_predictor_instance()
        m_name = pred.metadata.get("model_name", "Loaded")
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            model_loaded=True,
            model_name=m_name,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            version="2.0.0",
            model_loaded=False,
            model_name=None,
        )


@app.get("/model/metadata", tags=["Model Telemetry"])
def get_model_metadata():
    """Retrieve full model specifications, training timestamp, SHA-256 hash, and metrics."""
    pred = get_predictor_instance()
    return pred.metadata


@app.get("/model/features", tags=["Model Telemetry"])
def get_feature_importances():
    """Retrieve normalized feature importances and domain constraints."""
    pred = get_predictor_instance()
    return {
        "feature_importances": pred.get_feature_importances(),
        "allowed_localities": LOCALITIES,
        "allowed_furnished_statuses": FURNISHED_STATUSES,
    }


@app.post("/predict", response_model=ValuationResponse, tags=["Valuation"])
def predict_property_valuation(property_data: PropertyRequest):
    """
    Generate fair market valuation for a single residential property.
    Returns point estimate, 95% confidence interval, and ₹/sqft rate.
    """
    pred = get_predictor_instance()
    try:
        res = pred.predict(property_data.model_dump())
        return ValuationResponse(
            price_lakhs=res.price_lakhs,
            prediction_interval_lower=res.prediction_interval_lower,
            prediction_interval_upper=res.prediction_interval_upper,
            confidence_interval_lower=res.confidence_interval_lower,
            confidence_interval_upper=res.confidence_interval_upper,
            price_per_sqft_inr=res.price_per_sqft_inr,
            clamped=res.clamped,
            features_used=res.features_used,
        )
    except PredictionValidationError as pve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(pve)) from pve
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=BatchValuationResponse, tags=["Valuation"])
def predict_batch_valuations(batch_data: BatchValuationRequest):
    """
    Batch evaluate a portfolio of multiple properties.
    Computes total portfolio value and individual property breakdown.
    """
    pred = get_predictor_instance()
    results = []
    total_val = 0.0

    for prop in batch_data.properties:
        try:
            res = pred.predict(prop.model_dump())
            results.append(
                ValuationResponse(
                    price_lakhs=res.price_lakhs,
                    prediction_interval_lower=res.prediction_interval_lower,
                    prediction_interval_upper=res.prediction_interval_upper,
                    confidence_interval_lower=res.confidence_interval_lower,
                    confidence_interval_upper=res.confidence_interval_upper,
                    price_per_sqft_inr=res.price_per_sqft_inr,
                    clamped=res.clamped,
                    features_used=res.features_used,
                )
            )
            total_val += res.price_lakhs
        except PredictionValidationError as pve:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(pve)) from pve

    return BatchValuationResponse(
        total_properties=len(results),
        total_portfolio_value_lakhs=round(total_val, 2),
        results=results,
    )

"""
Central configuration for the House Price Prediction system.
Provides immutable paths, feature schemas, validation limits, and operational constants.
"""

from pathlib import Path
from typing import Dict, List, Any

# Filesystem layout
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

# Primary file paths
DATASET_PATH = DATA_DIR / "house_prices.csv"
CHAMPION_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_linear_model.pkl"
LEGACY_MODEL_PATH = ROOT_DIR / "house_price_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Features and Schema
LEGACY_FEATURES: List[str] = ["area", "bedrooms", "bathrooms", "age"]
NUMERIC_FEATURES: List[str] = ["area", "bedrooms", "bathrooms", "age", "parking"]
CATEGORICAL_FEATURES: List[str] = ["locality", "furnished_status"]
BINARY_FEATURES: List[str] = ["has_amenities"]
TARGET_COLUMN: str = "price"

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Feature categories and supported options
LOCALITIES: List[str] = ["Urban Prime", "Suburban", "Rural/Outskirts"]
FURNISHED_STATUSES: List[str] = ["Unfurnished", "Semi-Furnished", "Furnished"]

# Domain validation limits
FEATURE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "area": {"min": 100, "max": 15000, "type": (int, float), "default": 1500},
    "bedrooms": {"min": 1, "max": 12, "type": int, "default": 3},
    "bathrooms": {"min": 1, "max": 10, "type": int, "default": 2},
    "age": {"min": 0.0, "max": 100.0, "type": (int, float), "default": 5.0},
    "parking": {"min": 0, "max": 10, "type": int, "default": 1},
    "locality": {"allowed": LOCALITIES, "type": str, "default": "Suburban"},
    "furnished_status": {"allowed": FURNISHED_STATUSES, "type": str, "default": "Semi-Furnished"},
    "has_amenities": {"allowed": [0, 1], "type": int, "default": 1},
}

# Domain Business Rules
MIN_PRICE_FLOOR_LAKHS: float = 5.0
DEFAULT_CONFIDENCE_PERCENTILE: float = 0.95

# Training configuration
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

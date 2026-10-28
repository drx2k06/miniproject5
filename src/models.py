"""
Model definition, training harness, model comparison, and serialization with integrity hashing.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.config import (
    CHAMPION_MODEL_PATH,
    BASELINE_MODEL_PATH,
    LEGACY_MODEL_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    ALL_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessing import build_pipeline, create_preprocessor
from src.evaluation import calculate_metrics, evaluate_cross_validation, analyze_residuals


def get_candidate_models() -> Dict[str, Any]:
    """
    Return dictionary of candidate estimators with balanced default hyperparameters.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.05, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_split=4, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE
        ),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, str]:
    """
    Train all candidate models through preprocessing pipelines, compute cross-validation
    and test set metrics, and determine champion model.

    Returns:
        Tuple:
            - Dict of trained pipelines
            - Comparison results DataFrame
            - Name of the champion model
    """
    candidates = get_candidate_models()
    trained_pipelines: Dict[str, Pipeline] = {}
    comparison_records = []

    for name, estimator in candidates.items():
        pipeline = build_pipeline(regressor=estimator)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Holdout test metrics
        y_pred = pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred)

        # Cross-validation metrics
        cv_metrics = evaluate_cross_validation(pipeline, X_train, y_train)

        record = {
            "Model": name,
            "Test R2": test_metrics["R2"],
            "Test RMSE": test_metrics["RMSE"],
            "Test MAE": test_metrics["MAE"],
            "Test MAPE (%)": test_metrics["MAPE (%)"],
            "CV R2 (Mean)": round(cv_metrics["cv_r2_mean"], 4),
            "CV R2 (Std)": round(cv_metrics["cv_r2_std"], 4),
            "CV RMSE (Mean)": round(cv_metrics["cv_rmse_mean"], 4),
        }
        comparison_records.append(record)

    comparison_df = pd.DataFrame(comparison_records)
    # Model selection is strictly based on Cross-Validation on training data to prevent test-set leakage
    comparison_df.sort_values(by=["CV R2 (Mean)", "CV RMSE (Mean)"], ascending=[False, True], inplace=True)
    comparison_df.reset_index(drop=True, inplace=True)

    champion_name = comparison_df.iloc[0]["Model"]
    return trained_pipelines, comparison_df, champion_name


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model_artifacts(
    pipeline: Pipeline,
    model_name: str,
    test_metrics: Dict[str, float],
    cv_metrics: Dict[str, float],
    residual_stats: Dict[str, float],
    features: list,
    feature_importances: Optional[Dict[str, float]] = None,
    interval_coverage: Optional[Dict[str, float]] = None,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Save trained pipeline artifact and associated JSON metadata with SHA-256 integrity hash.
    """
    target_model_path = Path(model_path) if model_path else CHAMPION_MODEL_PATH
    target_meta_path = Path(metadata_path) if metadata_path else MODEL_METADATA_PATH

    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    target_meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pipeline binary
    joblib.dump(pipeline, target_model_path)
    file_sha256 = compute_sha256(target_model_path)

    # Also save to legacy root model path for backwards compatibility
    try:
        joblib.dump(pipeline, LEGACY_MODEL_PATH)
    except Exception:
        pass

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "algorithm": pipeline.named_steps["regressor"].__class__.__name__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_checksum": file_sha256,
        "features": features,
        "target": TARGET_COLUMN,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "residual_stats": residual_stats,
        "interval_coverage": interval_coverage or {},
        "feature_importances": feature_importances or {},
    }

    with open(target_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return target_model_path, target_meta_path


def load_model_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """
    Load saved model pipeline from disk with fallback hierarchy.
    """
    candidates = [
        Path(model_path) if model_path else None,
        CHAMPION_MODEL_PATH,
        LEGACY_MODEL_PATH,
        BASELINE_MODEL_PATH,
    ]

    for p in candidates:
        if p and p.exists():
            return joblib.load(p)

    raise FileNotFoundError(
        f"No trained model artifact found. Searched: {[str(p) for p in candidates if p]}"
    )

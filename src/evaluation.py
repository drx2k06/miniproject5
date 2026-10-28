"""
Evaluation metrics computation, cross-validation scoring, and residual analysis.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
)
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_STATE


def calculate_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate comprehensive regression evaluation metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict[str, float]: Dictionary containing computed metrics.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    medae = float(median_absolute_error(y_t, y_p))
    max_err = float(max_error(y_t, y_p))

    # Mean Absolute Percentage Error (avoid division by zero)
    non_zero_mask = y_t != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4),
        "Max_Error": round(max_err, 4),
    }


def analyze_residuals(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Compute statistical properties of model prediction residuals (y_true - y_pred).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict[str, float]: Mean, standard deviation, and percentiles of residuals.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "residual_p95": float(np.percentile(np.abs(residuals), 95)),
    }


def compute_interval_coverage(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    residual_std: float,
    z_score: float = 1.96,
) -> Dict[str, float]:
    """
    Compute Prediction Interval Coverage Probability (PICP) for uncertainty bounds.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        residual_std: Residual standard deviation.
        z_score: Z-multiplier for confidence level (1.96 for 95%).

    Returns:
        Dict[str, float]: Coverage percentage and bounds metrics.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    lower = y_p - (z_score * residual_std)
    upper = y_p + (z_score * residual_std)

    covered = (y_t >= lower) & (y_t <= upper)
    coverage_pct = float(np.mean(covered) * 100.0)

    return {
        "picp_coverage_pct": round(coverage_pct, 2),
        "mean_interval_width": round(float(np.mean(upper - lower)), 2),
        "target_coverage_pct": 95.0,
    }


def extract_feature_importances(pipeline: Pipeline) -> Dict[str, float]:
    """
    Extract normalized feature importances from a trained pipeline.
    Maps one-hot transformed column names back to clean, readable names.

    Args:
        pipeline: Trained Scikit-Learn Pipeline.

    Returns:
        Dict[str, float]: Feature name to relative importance percentage (0-100%).
    """
    if "preprocessor" not in pipeline.named_steps or "regressor" not in pipeline.named_steps:
        return {}

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    try:
        raw_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return {}

    # Extract raw importance or absolute coefficient
    if hasattr(regressor, "feature_importances_"):
        raw_weights = np.asarray(regressor.feature_importances_, dtype=float)
    elif hasattr(regressor, "coef_"):
        raw_weights = np.abs(np.asarray(regressor.coef_, dtype=float))
    else:
        return {}

    # Normalize to 100%
    total = np.sum(raw_weights)
    if total > 0:
        normalized = (raw_weights / total) * 100.0
    else:
        normalized = raw_weights

    # Clean display names
    clean_map = {
        "num__area": "Area (sq ft)",
        "num__bedrooms": "Bedrooms (BHK)",
        "num__bathrooms": "Bathrooms",
        "num__age": "Property Age (years)",
        "num__parking": "Parking Spaces",
        "cat__locality_Suburban": "Locality: Suburban",
        "cat__locality_Urban Prime": "Locality: Urban Prime",
        "cat__furnished_status_Semi-Furnished": "Furnished: Semi",
        "cat__furnished_status_Unfurnished": "Furnished: None",
        "bin__has_amenities": "Society Amenities",
    }

    importances = {}
    for name, weight in zip(raw_names, normalized):
        display_name = clean_map.get(name, name)
        importances[display_name] = float(weight)

    # Sort descending
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    rounded_items = {k: round(v, 2) for k, v in sorted_items}
    
    # Enforce exact 100.0% sum by adjusting rounding difference on primary feature
    diff = round(100.0 - sum(rounded_items.values()), 2)
    if diff != 0 and sorted_items:
        lead_feature = sorted_items[0][0]
        rounded_items[lead_feature] = round(rounded_items[lead_feature] + diff, 2)

    return rounded_items


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """
    Perform K-Fold Cross-Validation on a model pipeline and return aggregated metrics.

    Args:
        pipeline: Estimator or Pipeline.
        X: Feature matrix.
        y: Target series.
        n_splits: Number of CV folds.
        random_state: Random state for KFold shuffle.

    Returns:
        Dict[str, float]: Mean and std of CV metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }

    cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, return_train_score=False)

    return {
        "cv_r2_mean": float(np.mean(cv_results["test_r2"])),
        "cv_r2_std": float(np.std(cv_results["test_r2"])),
        "cv_mae_mean": float(-np.mean(cv_results["test_neg_mae"])),
        "cv_mae_std": float(np.std(cv_results["test_neg_mae"])),
        "cv_rmse_mean": float(-np.mean(cv_results["test_neg_rmse"])),
        "cv_rmse_std": float(np.std(cv_results["test_neg_rmse"])),
    }

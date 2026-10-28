"""
Standalone Evaluation & Model Diagnostics Script.
Computes comprehensive metrics, residual error diagnostics,
and physical consistency validation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure robust UTF-8 printing across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import load_dataset, split_dataset, get_baseline_10_row_data
from src.models import load_model_pipeline
from src.predictor import HousePricePredictor
from src.evaluation import calculate_metrics, analyze_residuals, compute_interval_coverage
from src.config import (
    CHAMPION_MODEL_PATH,
    BASELINE_MODEL_PATH,
    LEGACY_FEATURES,
    ALL_FEATURES,
)


def run_evaluation():
    print("=" * 75)
    print("  📊 House Price Prediction - Comprehensive Evaluation & Diagnostics")
    print("=" * 75)

    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    predictor = HousePricePredictor(model_path=CHAMPION_MODEL_PATH)

    # 1. Evaluate Champion Model on Holdout Test Set
    champion_preds = [predictor.predict(row.to_dict()).price_lakhs for _, row in X_test.iterrows()]
    champ_metrics = calculate_metrics(y_test, champion_preds)
    champ_residuals = analyze_residuals(y_test, champion_preds)

    print("\n[1] Champion Model Test Metrics (N = %d):" % len(y_test))
    for k, v in champ_metrics.items():
        print(f"    - {k:15s}: {v}")

    print("\n[2] Residual Diagnostics:")
    for k, v in champ_residuals.items():
        print(f"    - {k:15s}: {v:.4f}")

    cov = compute_interval_coverage(y_test, champion_preds, residual_std=champ_residuals["residual_std"])
    print("\n[2b] 95% Prediction Interval Calibration (Observation Uncertainty):")
    print(f"    - PICP Coverage  : {cov['picp_coverage_pct']}% (Target: 95.0%)")
    print(f"    - Mean Bandwidth : ₹{cov['mean_interval_width']} Lakhs (±₹{cov['mean_interval_width']/2:.2f}L)")

    feat_imps = predictor.get_feature_importances()
    if feat_imps:
        print("\n[2c] Feature Importance Drivers (Top 5):")
        for idx, (f_name, f_pct) in enumerate(list(feat_imps.items())[:5], 1):
            print(f"    {idx}. {f_name:25s}: {f_pct:.2f}%")

    # 2. Evaluate Baseline Linear Model
    base_metrics = None
    if BASELINE_MODEL_PATH.exists():
        import joblib
        base_pipe = joblib.load(BASELINE_MODEL_PATH)
        base_preds = base_pipe.predict(X_test[LEGACY_FEATURES])
        base_metrics = calculate_metrics(y_test, base_preds)
        print("\n[3] Baseline Linear Model Test Metrics (4 Features, N = %d):" % len(y_test))
        for k, v in base_metrics.items():
            print(f"    - {k:15s}: {v}")

    # 3. Physical Consistency Check
    print("\n[4] Physical Consistency & Domain Guardrails Audit:")
    
    # Test A: Tiny house negative price prevention
    tiny_house = {"area": 100, "bedrooms": 1, "bathrooms": 1, "age": 20}
    tiny_res = predictor.predict(tiny_house)
    print(f"    - Test A (100 sqft extreme edge case):")
    print(f"      Predicted: ₹{tiny_res.price_lakhs} Lakhs (Clamped floor: {tiny_res.clamped})")
    assert tiny_res.price_lakhs > 0, "Price must be strictly positive!"

    # Test B: Bedroom marginal valuation (monotonicity)
    p_3bhk = predictor.predict({"area": 1800, "bedrooms": 3, "bathrooms": 2, "age": 5}).price_lakhs
    p_4bhk = predictor.predict({"area": 1800, "bedrooms": 4, "bathrooms": 2, "age": 5}).price_lakhs
    print(f"    - Test B (Bedroom Marginal Utility, 1800 sqft):")
    print(f"      3 BHK: ₹{p_3bhk} Lakhs | 4 BHK: ₹{p_4bhk} Lakhs (Diff: +₹{p_4bhk - p_3bhk:.2f} Lakhs)")

    # Test C: Locality Premium
    p_prime = predictor.predict({"area": 1800, "bedrooms": 3, "bathrooms": 2, "age": 5, "locality": "Urban Prime"}).price_lakhs
    p_suburb = predictor.predict({"area": 1800, "bedrooms": 3, "bathrooms": 2, "age": 5, "locality": "Suburban"}).price_lakhs
    p_rural = predictor.predict({"area": 1800, "bedrooms": 3, "bathrooms": 2, "age": 5, "locality": "Rural/Outskirts"}).price_lakhs
    print(f"    - Test C (Locality Impact, 1800 sqft):")
    print(f"      Urban Prime: ₹{p_prime}L | Suburban: ₹{p_suburb}L | Rural: ₹{p_rural}L")

    print("\n" + "=" * 75)
    print("  Evaluation and Diagnostics completed successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_evaluation()

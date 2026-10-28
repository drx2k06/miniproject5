"""
CLI Training Script for House Price Prediction.
Trains multiple candidate algorithms, evaluates on holdout and cross-validation folds,
selects the champion model, and serializes artifacts with integrity metadata.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure robust UTF-8 printing across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import load_dataset, split_dataset
from src.models import (
    train_and_compare_models,
    save_model_artifacts,
    build_pipeline,
)
from src.preprocessing import create_legacy_preprocessor
from src.evaluation import (
    calculate_metrics,
    analyze_residuals,
    evaluate_cross_validation,
    compute_interval_coverage,
    extract_feature_importances,
)
from src.config import (
    CHAMPION_MODEL_PATH,
    BASELINE_MODEL_PATH,
    MODEL_METADATA_PATH,
    ALL_FEATURES,
    LEGACY_FEATURES,
)
from sklearn.linear_model import LinearRegression


def run_training():
    print("=" * 70)
    print("  🏠 House Price Prediction - Model Training & Evaluation Harness")
    print("=" * 70)

    # 1. Ingestion
    print("\n[1/5] Loading and validating dataset...")
    df = load_dataset()
    print(f"      Loaded {len(df)} validated property records.")

    # 2. Partition
    print("\n[2/5] Partitioning data into train and test splits (stratified by price)...")
    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"      Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Model Comparison
    print("\n[3/5] Training and benchmarking candidate algorithms...")
    trained_pipelines, comparison_df, champion_name = train_and_compare_models(
        X_train, y_train, X_test, y_test
    )

    print("\n" + "-" * 70)
    print("  Model Comparison Benchmark Results (Sorted by CV R2 - Leak-Free Selection)")
    print("-" * 70)
    print(comparison_df.to_string(index=False))
    print("-" * 70)
    print(f"🏆 Champion Model Selected: {champion_name}")

    # 4. Champion Artifact Serialization
    champion_pipeline = trained_pipelines[champion_name]
    y_test_pred = champion_pipeline.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    cv_metrics = evaluate_cross_validation(champion_pipeline, X_train, y_train)
    residual_stats = analyze_residuals(y_test, y_test_pred)
    feat_importances = extract_feature_importances(champion_pipeline)
    interval_cov = compute_interval_coverage(
        y_test, y_test_pred, residual_std=residual_stats["residual_std"]
    )

    print("\n[4/5] Serializing champion model and JSON metadata...")
    model_p, meta_p = save_model_artifacts(
        pipeline=champion_pipeline,
        model_name=champion_name,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
        residual_stats=residual_stats,
        features=ALL_FEATURES,
        feature_importances=feat_importances,
        interval_coverage=interval_cov,
        model_path=CHAMPION_MODEL_PATH,
        metadata_path=MODEL_METADATA_PATH,
    )
    print(f"      Saved champion pipeline to: {model_p}")
    print(f"      Saved model metadata to:    {meta_p}")
    print(f"      Empirical 95% Prediction Interval Coverage (PICP): {interval_cov['picp_coverage_pct']}%")

    # 5. Baseline Reference Model (for comparative audit)
    print("\n[5/5] Training and saving baseline reference Linear Regression model...")
    legacy_prep = create_legacy_preprocessor()
    base_pipe = build_pipeline(regressor=LinearRegression(), preprocessor=legacy_prep)
    base_pipe.fit(X_train[LEGACY_FEATURES], y_train)
    import joblib
    joblib.dump(base_pipe, BASELINE_MODEL_PATH)
    base_test_pred = base_pipe.predict(X_test[LEGACY_FEATURES])
    base_metrics = calculate_metrics(y_test, base_test_pred)
    print(f"      Baseline Linear Model saved to: {BASELINE_MODEL_PATH}")
    print(f"      Baseline Metrics on Benchmark: R2={base_metrics['R2']}, RMSE={base_metrics['RMSE']}, MAE={base_metrics['MAE']}")

    print("\n" + "=" * 70)
    print(f"  Training workflow completed successfully! Champion: {champion_name}")
    print("=" * 70 + "\n")
    return comparison_df


if __name__ == "__main__":
    run_training()

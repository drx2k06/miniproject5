"""Unit tests for src/models.py."""

import json
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from src.data_loader import load_dataset, split_dataset
from src.models import (
    get_candidate_models,
    train_and_compare_models,
    save_model_artifacts,
    load_model_pipeline,
    compute_sha256,
)
from src.preprocessing import build_pipeline
from src.config import ALL_FEATURES


def test_get_candidate_models():
    """Verify all 5 candidate models are provided."""
    models = get_candidate_models()
    expected = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "Gradient Boosting"]
    for name in expected:
        assert name in models


def test_train_and_compare_models():
    """Verify model comparison workflow and sorting by CV R2 (leak-free selection)."""
    df = load_dataset().head(100)  # Use subset for fast test
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.25, random_state=42)

    pipelines, comparison_df, champion_name = train_and_compare_models(
        X_train, y_train, X_test, y_test
    )

    assert len(pipelines) == 5
    assert len(comparison_df) == 5
    assert champion_name in pipelines
    assert "Test R2" in comparison_df.columns
    assert "Test RMSE" in comparison_df.columns
    assert "CV R2 (Mean)" in comparison_df.columns
    # Ensure descending sort by CV R2 (Mean)
    assert comparison_df["CV R2 (Mean)"].iloc[0] >= comparison_df["CV R2 (Mean)"].iloc[-1]


def test_save_and_load_model_artifacts(tmp_path: Path):
    """Verify artifact serialization, sha256 generation, and reloading."""
    df = load_dataset().head(60)
    X = df[ALL_FEATURES]
    y = df["price"]

    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)

    model_p = tmp_path / "test_model.pkl"
    meta_p = tmp_path / "test_meta.json"

    save_model_artifacts(
        pipeline=pipeline,
        model_name="TestLinear",
        test_metrics={"MAE": 5.0, "RMSE": 7.0, "R2": 0.90},
        cv_metrics={"cv_r2_mean": 0.89},
        residual_stats={"residual_std": 6.8},
        features=ALL_FEATURES,
        model_path=model_p,
        metadata_path=meta_p,
    )

    assert model_p.exists()
    assert meta_p.exists()

    # Verify sha256 in metadata matches file
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["sha256_checksum"] == compute_sha256(model_p)

    # Load and test prediction
    loaded = load_model_pipeline(model_p)
    assert isinstance(loaded, Pipeline)
    preds = loaded.predict(X.head(5))
    assert len(preds) == 5

"""Unit tests for src/preprocessing.py."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.preprocessing import create_preprocessor, create_legacy_preprocessor, build_pipeline
from src.config import ALL_FEATURES, LEGACY_FEATURES


def test_create_preprocessor():
    """Verify that full preprocessor fits and transforms without error."""
    df = load_dataset()
    X = df[ALL_FEATURES]

    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(X)
    # 5 numerics + 2 dummy categories for locality + 2 for furnished + 1 binary = 10 cols
    assert X_transformed.shape[1] >= 9


def test_create_legacy_preprocessor():
    """Verify legacy 4-feature preprocessor."""
    df = load_dataset()
    X_legacy = df[LEGACY_FEATURES]

    legacy_prep = create_legacy_preprocessor()
    X_trans = legacy_prep.fit_transform(X_legacy)

    assert X_trans.shape == (len(df), 4)


def test_build_pipeline():
    """Verify that end-to-end pipeline executes fit and predict cleanly."""
    df = load_dataset()
    X = df[ALL_FEATURES].head(50)
    y = df["price"].head(50)

    pipeline = build_pipeline(regressor=LinearRegression())
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert isinstance(pipeline, Pipeline)

"""Unit tests for src/data_loader.py."""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    load_dataset,
    split_dataset,
    get_baseline_10_row_data,
    DataValidationError,
)
from src.config import ALL_FEATURES, TARGET_COLUMN


def test_load_dataset_success():
    """Verify that the benchmark dataset loads cleanly with proper schema."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns
    assert (df["area"] > 0).all()
    assert (df["bedrooms"] > 0).all()
    assert (df["bathrooms"] > 0).all()
    assert (df["age"] >= 0).all()
    assert (df["price"] > 0).all()


def test_load_dataset_missing_file(tmp_path: Path):
    """Verify FileNotFoundError on non-existent dataset path."""
    fake_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(fake_path)


def test_load_dataset_invalid_columns(tmp_path: Path):
    """Verify DataValidationError when required columns are missing."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"area": [1000], "price": [50]}).to_csv(bad_csv, index=False)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_dataset(bad_csv)


def test_load_dataset_negative_area(tmp_path: Path):
    """Verify DataValidationError when negative area exists."""
    df = load_dataset().copy()
    df.loc[0, "area"] = -500
    bad_csv = tmp_path / "neg_area.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(DataValidationError, match="non-positive values in 'area'"):
        load_dataset(bad_csv)


def test_split_dataset():
    """Verify train/test split proportions and target partition."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.20, random_state=42)

    total_len = len(df)
    assert len(X_train) == int(total_len * 0.8)
    assert len(X_test) == total_len - len(X_train)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Ensure no index intersection between train and test
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0


def test_baseline_10_row_data():
    """Verify the original 10-row baseline dataset."""
    df_base = get_baseline_10_row_data()
    assert len(df_base) == 10
    assert list(df_base.columns) == ["area", "bedrooms", "bathrooms", "age", "price"]

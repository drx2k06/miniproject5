"""
Data loading and dataset validation pipeline.
Handles file ingestion, schema validation, physical constraints checking,
and stratified dataset splitting.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    ALL_FEATURES,
    TARGET_COLUMN,
    FEATURE_CONSTRAINTS,
    RANDOM_STATE,
    TEST_SIZE,
)


class DataValidationError(ValueError):
    """Raised when dataset does not satisfy schema or integrity constraints."""
    pass


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Safely load housing dataset from CSV with schema and data integrity verification.

    Args:
        filepath: Optional Path to dataset CSV. Defaults to config.DATASET_PATH.

    Returns:
        pd.DataFrame: Clean validated DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        DataValidationError: If schema or value constraints are violated.
    """
    target_path = Path(filepath) if filepath else DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")

    try:
        df = pd.read_csv(target_path)
    except Exception as exc:
        raise DataValidationError(f"Failed to read CSV at {target_path}: {exc}") from exc

    if df.empty:
        raise DataValidationError("Dataset is empty.")

    # Verify column presence
    expected_cols = set(ALL_FEATURES + [TARGET_COLUMN])
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns in dataset: {sorted(missing_cols)}")

    # Check missing values
    null_counts = df[list(expected_cols)].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise DataValidationError(f"Null values detected in dataset: {cols_with_nulls}")

    # Validate physical constraints
    if (df["area"] <= 0).any():
        raise DataValidationError("Found non-positive values in 'area'.")
    if (df["bedrooms"] <= 0).any():
        raise DataValidationError("Found non-positive values in 'bedrooms'.")
    if (df["bathrooms"] <= 0).any():
        raise DataValidationError("Found non-positive values in 'bathrooms'.")
    if (df["age"] < 0).any():
        raise DataValidationError("Found negative values in 'age'.")
    if (df[TARGET_COLUMN] <= 0).any():
        raise DataValidationError("Found non-positive values in target 'price'.")

    # Validate categorical levels
    valid_localities = set(FEATURE_CONSTRAINTS["locality"]["allowed"])
    invalid_loc = set(df["locality"].unique()) - valid_localities
    if invalid_loc:
        raise DataValidationError(f"Invalid locality categories found: {invalid_loc}")

    valid_furnished = set(FEATURE_CONSTRAINTS["furnished_status"]["allowed"])
    invalid_furn = set(df["furnished_status"].unique()) - valid_furnished
    if invalid_furn:
        raise DataValidationError(f"Invalid furnished_status categories found: {invalid_furn}")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify_by_price: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition dataset into training and test splits.
    Optionally stratifies across price quintiles to maintain consistent target distribution.

    Args:
        df: Input DataFrame.
        test_size: Fraction of samples to allocate to test set.
        random_state: Seed for reproducibility.
        stratify_by_price: Whether to stratify split using price quintile bins.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    stratify_labels = None
    if stratify_by_price and len(df) >= 20:
        # Create 5 quantile bins for stratified sampling
        try:
            stratify_labels = pd.qcut(y, q=5, labels=False, duplicates="drop")
        except ValueError:
            stratify_labels = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return X_train, X_test, y_train, y_test


def get_baseline_10_row_data() -> pd.DataFrame:
    """
    Returns the exact 10-row dataset from the original baseline repository for verification and audit.
    """
    return pd.DataFrame({
        "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
        "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        "bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "age": [20, 15, 10, 8, 5, 7, 4, 3, 2, 1],
        "price": [25, 30, 40, 48, 55, 62, 72, 82, 95, 110],
    })

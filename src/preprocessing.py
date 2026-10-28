"""
Feature preprocessing pipelines and column transformation builders.
Provides clean feature transformers preventing data leakage between splits.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    LEGACY_FEATURES,
)


def create_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that scales numerical variables, one-hot encodes
    categoricals, and passes through binary flags.

    Args:
        numeric_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        binary_cols: List of binary column names.

    Returns:
        ColumnTransformer: Configured transformer.
    """
    nums = numeric_cols if numeric_cols is not None else NUMERIC_FEATURES
    cats = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES
    bins = binary_cols if binary_cols is not None else BINARY_FEATURES

    transformers = []

    if nums:
        transformers.append(
            ("num", StandardScaler(), nums)
        )
    if cats:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cats)
        )
    if bins:
        transformers.append(
            ("bin", "passthrough", bins)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def create_legacy_preprocessor() -> ColumnTransformer:
    """
    Construct a preprocessor for the original 4-feature schema (area, bedrooms, bathrooms, age).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), LEGACY_FEATURES)
        ],
        remainder="drop",
    )


def build_pipeline(regressor: BaseEstimator, preprocessor: Optional[ColumnTransformer] = None) -> Pipeline:
    """
    Attach an estimator to a feature preprocessing transformer within a Scikit-Learn Pipeline.

    Args:
        regressor: A scikit-learn compatible regression estimator.
        preprocessor: Optional ColumnTransformer. Defaults to create_preprocessor().

    Returns:
        Pipeline: Combined preprocessing and estimation pipeline.
    """
    prep = preprocessor if preprocessor is not None else create_preprocessor()
    return Pipeline(steps=[
        ("preprocessor", prep),
        ("regressor", regressor),
    ])

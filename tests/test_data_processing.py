"""
Unit tests for data processing and pipeline preprocessing.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.data_processing import (
    load_raw_data,
    validate_raw_data,
    split_data,
    get_preprocessor,
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    BINARY_FEATURES,
    NUMERICAL_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
)


def test_raw_data_exists_and_shape():
    """Verify raw dataset exists and matches expected 4424 x 37 dimensions."""
    assert os.path.exists(RAW_DATA_PATH), f"Raw data file {RAW_DATA_PATH} not found"
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.shape == (4424, 37), f"Expected (4424, 37), got {df.shape}"


def test_no_missing_values_in_raw():
    """Verify 0 missing values in the raw dataset."""
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.isnull().sum().sum() == 0, "Expected 0 missing values in raw dataset"


def test_no_duplicate_rows():
    """Verify 0 duplicate rows in raw dataset."""
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.duplicated().sum() == 0, "Expected 0 duplicate rows"


def test_target_classes():
    """Verify the 3 target classes: Dropout, Enrolled, Graduate."""
    df = pd.read_csv(RAW_DATA_PATH)
    classes = set(df[TARGET_COLUMN].unique())
    assert classes == {"Dropout", "Enrolled", "Graduate"}


def test_stratified_split_integrity():
    """Verify train/test split has exact 80/20 ratio and zero index leakage."""
    df = pd.read_csv(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    
    assert len(X_train) == 3539, f"Expected 3539 train rows, got {len(X_train)}"
    assert len(X_test) == 885, f"Expected 885 test rows, got {len(X_test)}"
    assert len(X_train) + len(X_test) == 4424

    # No overlapping index
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0

    # Stratification check (class proportions within 1% of raw)
    for cls in ["Dropout", "Enrolled", "Graduate"]:
        raw_prop = (df[TARGET_COLUMN] == cls).mean()
        train_prop = (y_train == cls).mean()
        test_prop = (y_test == cls).mean()
        assert abs(raw_prop - train_prop) < 0.01
        assert abs(raw_prop - test_prop) < 0.01


def test_preprocessor_transformation():
    """Verify ColumnTransformer transforms features without NaNs."""
    df = pd.read_csv(RAW_DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    preprocessor = get_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    
    assert isinstance(X_trans, np.ndarray)
    assert not np.isnan(X_trans).any(), "Found NaNs in transformed feature matrix"
    assert X_trans.shape[0] == 4424
    # Transformed features count should be > 36 due to one-hot encoding
    assert X_trans.shape[1] > 36

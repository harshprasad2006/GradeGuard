"""
Comprehensive unit tests for data processing, validation, feature taxonomy, and pipelines.
"""

import os
import sys
import json
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
    METADATA_PATH,
    CLASS_MAPPING_PATH,
    TARGET_COLUMN,
    BINARY_FEATURES,
    NUMERICAL_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def test_raw_data_exists():
    """Verify raw dataset exists on disk."""
    assert os.path.exists(RAW_DATA_PATH), f"Raw data file {RAW_DATA_PATH} not found"


def test_raw_data_dimensions():
    """Verify raw dataset matches official 4,424 rows and 37 columns."""
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.shape == (4424, 37), f"Expected (4424, 37), got {df.shape}"


def test_no_missing_values_in_raw():
    """Verify 0 missing values across all columns."""
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.isnull().sum().sum() == 0, "Expected 0 missing values"


def test_no_duplicate_rows():
    """Verify 0 duplicate records."""
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.duplicated().sum() == 0, "Expected 0 duplicate rows"


def test_target_classes_exact_match():
    """Verify target classes are Dropout, Enrolled, Graduate."""
    df = pd.read_csv(RAW_DATA_PATH)
    classes = set(df[TARGET_COLUMN].unique())
    assert classes == {"Dropout", "Enrolled", "Graduate"}


def test_feature_counts_taxonomy():
    """Verify taxonomy counts: 8 binary, 19 numerical, 9 nominal = 36 features."""
    assert len(BINARY_FEATURES) == 8, f"Expected 8 binary, got {len(BINARY_FEATURES)}"
    assert len(NUMERICAL_FEATURES) == 19, f"Expected 19 numerical, got {len(NUMERICAL_FEATURES)}"
    assert len(NOMINAL_CATEGORICAL_FEATURES) == 9, f"Expected 9 nominal, got {len(NOMINAL_CATEGORICAL_FEATURES)}"
    total_features = len(BINARY_FEATURES) + len(NUMERICAL_FEATURES) + len(NOMINAL_CATEGORICAL_FEATURES)
    assert total_features == 36


def test_all_features_present_in_dataset():
    """Verify every defined feature exists in the raw CSV."""
    df = pd.read_csv(RAW_DATA_PATH)
    all_expected = BINARY_FEATURES + NUMERICAL_FEATURES + NOMINAL_CATEGORICAL_FEATURES
    for feat in all_expected:
        assert feat in df.columns, f"Feature '{feat}' missing from dataset"


def test_binary_features_strictly_boolean():
    """Verify binary features contain only values in {0, 1}."""
    df = pd.read_csv(RAW_DATA_PATH)
    for col in BINARY_FEATURES:
        unique_vals = set(df[col].dropna().unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} has non-binary values: {unique_vals}"


def test_numerical_ranges_bounds():
    """Verify continuous academic and demographic features fall within expected limits."""
    df = pd.read_csv(RAW_DATA_PATH)
    assert (df["Admission grade"] >= 0).all() and (df["Admission grade"] <= 200).all()
    assert (df["Previous qualification (grade)"] >= 0).all() and (df["Previous qualification (grade)"] <= 200).all()
    assert (df["Age at enrollment"] >= 17).all() and (df["Age at enrollment"] <= 75).all()
    assert (df["Curricular units 1st sem (grade)"] >= 0).all() and (df["Curricular units 1st sem (grade)"] <= 20).all()
    assert (df["Curricular units 2nd sem (grade)"] >= 0).all() and (df["Curricular units 2nd sem (grade)"] <= 20).all()


def test_stratified_split_row_counts():
    """Verify 80% train (3,539) and 20% test (885) row counts."""
    df = pd.read_csv(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    assert len(X_train) == 3539
    assert len(X_test) == 885
    assert len(X_train) + len(X_test) == 4424


def test_zero_index_leakage_between_splits():
    """Verify complete index disjointness between train and test sets."""
    df = pd.read_csv(RAW_DATA_PATH)
    X_train, X_test, _, _ = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    overlap = set(X_train.index).intersection(set(X_test.index))
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"


def test_stratification_class_proportions():
    """Verify class proportions match between raw, train, and test partitions."""
    df = pd.read_csv(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for cls in ["Dropout", "Enrolled", "Graduate"]:
        raw_p = (df[TARGET_COLUMN] == cls).mean()
        train_p = (y_train == cls).mean()
        test_p = (y_test == cls).mean()
        assert abs(raw_p - train_p) < 0.005, f"Train proportion drift for {cls}"
        assert abs(raw_p - test_p) < 0.005, f"Test proportion drift for {cls}"


def test_processed_files_exist_on_disk():
    """Verify saved train and test processed CSVs exist."""
    assert os.path.exists(TRAIN_DATA_PATH)
    assert os.path.exists(TEST_DATA_PATH)


def test_preprocessor_transformation_matrix():
    """Verify ColumnTransformer outputs valid dense array without NaNs or Infs."""
    df = pd.read_csv(RAW_DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    preprocessor = get_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    
    assert isinstance(X_trans, np.ndarray)
    assert not np.isnan(X_trans).any(), "Found NaNs in transformed feature matrix"
    assert not np.isinf(X_trans).any(), "Found Infs in transformed feature matrix"
    assert X_trans.shape[0] == 4424
    # With one-hot encoding, feature dimension expands beyond 36
    assert X_trans.shape[1] > 36


def test_feature_metadata_json_schema():
    """Verify feature metadata JSON contains required keys."""
    assert os.path.exists(METADATA_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["dataset_id"] == 697
    assert meta["total_instances"] == 4424
    assert meta["total_features"] == 36
    assert "feature_ranges" in meta
    assert "human_readable_labels" in meta


def test_class_mapping_json_bidirectional():
    """Verify class mapping JSON maps integer to string consistently."""
    assert os.path.exists(CLASS_MAPPING_PATH)
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    t2i = mapping["target_to_int"]
    i2t = mapping["int_to_target"]
    for label, idx in t2i.items():
        assert i2t[str(idx)] == label

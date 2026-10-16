"""
Data Processing Module for GradeGuard.
Handles data loading, validation, feature categorization, stratified splitting,
and reproducible scikit-learn preprocessing pipelines without data leakage.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_PATH = os.path.join("data", "raw", "dataset.csv")
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train_data.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "test_data.csv")
METADATA_PATH = os.path.join(PROCESSED_DIR, "feature_metadata.json")
CLASS_MAPPING_PATH = os.path.join(PROCESSED_DIR, "class_mapping.json")

TARGET_COLUMN = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature Taxonomy based on UCI ID 697 and Realinho et al. (2021)
BINARY_FEATURES: List[str] = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERICAL_FEATURES: List[str] = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

NOMINAL_CATEGORICAL_FEATURES: List[str] = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Human-readable dictionary labels for UI presentation
HUMAN_READABLE_LABELS: Dict[str, Dict[str, Any]] = {
    "Marital Status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto Union",
        6: "Legally Separated",
    },
    "Application mode": {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores Island)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira Island)",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
        27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)",
    },
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening attendance)",
    },
    "Daytime/evening attendance": {1: "Daytime", 0: "Evening"},
    "Previous qualification": {
        1: "Secondary education",
        2: "Higher education - bachelor's degree",
        3: "Higher education - degree",
        4: "Higher education - master's",
        5: "Higher education - doctorate",
        6: "Frequency of higher education",
        9: "12th year of schooling - not completed",
        10: "11th year of schooling - not completed",
        12: "Other - 11th year of schooling",
        14: "10th year of schooling",
        15: "10th year of schooling - not completed",
        19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        39: "Technological specialization course",
        40: "Higher education - degree (1st cycle)",
        42: "Professional higher technical course",
        43: "Higher education - master (2nd cycle)",
    },
    "Nacionality": {
        1: "Portuguese",
        2: "German",
        6: "Spanish",
        11: "Italian",
        13: "Dutch",
        14: "English",
        17: "Lithuanian",
        21: "Angolan",
        22: "Cape Verdean",
        24: "Guinean",
        25: "Mozambican",
        26: "Santomean",
        32: "Turkish",
        41: "Brazilian",
        62: "Romanian",
        100: "Moldova (Republic of)",
        101: "Mexican",
        103: "Ukrainian",
        105: "Russian",
        108: "Cuban",
        109: "Colombian",
    },
    "Displaced": {1: "Yes", 0: "No"},
    "Educational special needs": {1: "Yes", 0: "No"},
    "Debtor": {1: "Yes", 0: "No"},
    "Tuition fees up to date": {1: "Yes", 0: "No"},
    "Gender": {1: "Male", 0: "Female"},
    "Scholarship holder": {1: "Yes", 0: "No"},
    "International": {1: "Yes", 0: "No"},
}

TARGET_ENCODING: Dict[str, int] = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2,
}
INVERSE_TARGET_ENCODING: Dict[int, str] = {v: k for k, v in TARGET_ENCODING.items()}


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV or download from UCI repository if missing."""
    if not os.path.exists(filepath):
        logger.info(f"Raw data file not found at {filepath}. Attempting UCI download...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=697)
            X = dataset.data.features
            y = dataset.data.targets
            df = pd.concat([X, y], axis=1)
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully downloaded and saved dataset to {filepath}")
        except Exception as exc:
            raise FileNotFoundError(
                f"Dataset not found at {filepath} and automated download failed: {exc}\n"
                "Please place official dataset.csv into data/raw/ to continue."
            ) from exc

    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw dataset from {filepath} with shape {df.shape}")
    return df


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect and validate data properties: shape, dtypes, missingness, duplicates, target."""
    issues = []
    
    # Check shape
    expected_rows = 4424
    expected_cols = 37  # 36 features + 1 Target
    if df.shape[1] != expected_cols:
        issues.append(f"Unexpected column count: {df.shape[1]} (expected {expected_cols})")
        
    # Check target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset!")
        
    missing_count = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    
    # Verify all 36 expected features
    all_expected = BINARY_FEATURES + NUMERICAL_FEATURES + NOMINAL_CATEGORICAL_FEATURES
    missing_features = [f for f in all_expected if f not in df.columns]
    if missing_features:
        issues.append(f"Missing expected features: {missing_features}")
        
    class_dist = df[TARGET_COLUMN].value_counts().to_dict()
    
    validation_report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": missing_count,
        "duplicate_rows": dup_count,
        "class_distribution": class_dist,
        "issues": issues,
        "is_valid": len(issues) == 0,
    }
    logger.info(f"Validation Report: {validation_report}")
    return validation_report


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified 80/20 train/test split.
    Guarantees no leakage between training and testing sets.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"Split data with random_state={random_state}: "
        f"Train={X_train.shape[0]} rows (80%), Test={X_test.shape[0]} rows (20%)"
    )
    return X_train, X_test, y_train, y_test


def get_preprocessor() -> ColumnTransformer:
    """
    Create a leak-free scikit-learn ColumnTransformer.
    - Numerical: SimpleImputer(median) -> StandardScaler()
    - Binary: SimpleImputer(most_frequent) -> 'passthrough'
    - Nominal Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERICAL_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat", nominal_pipe, NOMINAL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def save_processed_data():
    """Load raw data, perform stratified split, save datasets and metadata."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_raw_data()
    validation = validate_raw_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save train and test sets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info(f"Saved processed train data to {TRAIN_DATA_PATH}")
    logger.info(f"Saved processed test data to {TEST_DATA_PATH}")

    # Build and save feature metadata
    metadata = {
        "dataset_name": "Predict Students' Dropout and Academic Success",
        "dataset_id": 697,
        "doi": "10.24432/C5MC89",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_instances": int(df.shape[0]),
        "total_features": len(X_train.columns),
        "target_column": TARGET_COLUMN,
        "binary_features": BINARY_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "nominal_categorical_features": NOMINAL_CATEGORICAL_FEATURES,
        "all_features_in_order": list(X_train.columns),
        "class_labels": list(TARGET_ENCODING.keys()),
        "target_encoding": TARGET_ENCODING,
        "validation_report": validation,
        "human_readable_labels": HUMAN_READABLE_LABELS,
        "feature_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
            }
            for col in NUMERICAL_FEATURES
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature metadata to {METADATA_PATH}")

    with open(CLASS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_to_int": TARGET_ENCODING,
                "int_to_target": INVERSE_TARGET_ENCODING,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved class mapping to {CLASS_MAPPING_PATH}")


if __name__ == "__main__":
    save_processed_data()

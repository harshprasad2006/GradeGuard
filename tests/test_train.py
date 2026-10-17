"""
Comprehensive unit tests for model training artifacts, pipelines, and cross-validation integrity.
"""

import os
import sys
import json
import pytest
import joblib
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.train import define_models_and_grids
from src.data_processing import TRAIN_DATA_PATH, TARGET_COLUMN

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def test_best_model_file_exists():
    """Verify best_model.joblib exists."""
    assert os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib"))


def test_all_six_model_files_exist():
    """Verify all 6 classical ML pipeline artifacts exist."""
    models = [
        "random_forest.joblib",
        "logistic_regression.joblib",
        "support_vector_machine.joblib",
        "decision_tree.joblib",
        "k_nearest_neighbours.joblib",
        "gaussian_naive_bayes.joblib",
    ]
    for m in models:
        path = os.path.join(MODELS_DIR, m)
        assert os.path.exists(path), f"Artifact {m} missing"


def test_model_metadata_structure():
    """Verify model_metadata.json structure and keys."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "best_model_name" in meta
    assert "selection_metric" in meta
    assert "best_cv_macro_f1" in meta
    assert meta["best_cv_macro_f1"] > 0.65
    assert len(meta["cv_results"]) == 6


def test_pipeline_composition_best_model():
    """Verify best model is a Pipeline containing preprocessor and model steps."""
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    assert hasattr(pipeline, "named_steps")
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_composition_all_models():
    """Verify every saved model has preprocessor and model steps."""
    models = [
        "random_forest.joblib",
        "logistic_regression.joblib",
        "support_vector_machine.joblib",
        "decision_tree.joblib",
        "k_nearest_neighbours.joblib",
        "gaussian_naive_bayes.joblib",
    ]
    for m in models:
        p = joblib.load(os.path.join(MODELS_DIR, m))
        assert "preprocessor" in p.named_steps
        assert "model" in p.named_steps


def test_model_definitions_dictionary():
    """Verify define_models_and_grids returns all 6 algorithms."""
    grids = define_models_and_grids()
    assert len(grids) == 6
    expected_names = {
        "Logistic Regression",
        "K-Nearest Neighbours",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "Gaussian Naive Bayes",
    }
    assert set(grids.keys()) == expected_names


def test_class_weighting_exploration():
    """Verify class_weight parameter is present in grids for supporting algorithms."""
    grids = define_models_and_grids()
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]:
        _, param_grid = grids[name]
        assert "model__class_weight" in param_grid, f"Missing class_weight in {name} grid"


def test_best_model_predict_smoke_test():
    """Verify best model executes prediction on sample training record."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    sample_X = train_df.drop(columns=[TARGET_COLUMN]).head(5)
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    preds = pipeline.predict(sample_X)
    assert len(preds) == 5
    for p in preds:
        assert p in [0, 1, 2]


def test_best_model_predict_proba_smoke_test():
    """Verify best model outputs 3 class probabilities summing to 1."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    sample_X = train_df.drop(columns=[TARGET_COLUMN]).head(3)
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    probs = pipeline.predict_proba(sample_X)
    assert probs.shape == (3, 3)
    for row in probs:
        assert abs(row.sum() - 1.0) < 1e-4


def test_logistic_regression_predict_proba():
    """Verify Logistic Regression outputs valid probabilities."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    sample_X = train_df.drop(columns=[TARGET_COLUMN]).head(2)
    pipeline = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    probs = pipeline.predict_proba(sample_X)
    assert probs.shape == (2, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

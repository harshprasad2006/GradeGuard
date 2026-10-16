"""
Unit tests for model artifact verification and pipeline integrity.
"""

import os
import sys
import json
import pytest
import joblib

sys.path.insert(0, os.path.abspath("."))
from src.train import define_models_and_grids

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def test_model_files_exist():
    """Verify that all 6 trained model pipeline artifacts exist on disk."""
    expected_models = [
        "best_model.joblib",
        "random_forest.joblib",
        "logistic_regression.joblib",
        "support_vector_machine.joblib",
        "decision_tree.joblib",
        "k_nearest_neighbours.joblib",
        "gaussian_naive_bayes.joblib",
    ]
    for model_file in expected_models:
        path = os.path.join(MODELS_DIR, model_file)
        assert os.path.exists(path), f"Expected model artifact {path} does not exist"


def test_model_metadata_validity():
    """Verify that model_metadata.json is populated with best model and CV results."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "best_model_name" in meta
    assert "best_cv_macro_f1" in meta
    assert meta["best_cv_macro_f1"] > 0.60
    assert len(meta["cv_results"]) == 6


def test_pipeline_structure():
    """Verify best model is a scikit-learn Pipeline with preprocessor and model steps."""
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    pipeline = joblib.load(best_path)
    assert hasattr(pipeline, "named_steps")
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

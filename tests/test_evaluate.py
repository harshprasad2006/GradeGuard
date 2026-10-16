"""
Unit tests for evaluation metrics and reporting artifacts.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

REPORTS_DIR = "reports"
METRICS_CSV = os.path.join(REPORTS_DIR, "metrics_summary.csv")
CV_CSV = os.path.join(REPORTS_DIR, "cv_results.csv")
EVAL_JSON = os.path.join(REPORTS_DIR, "evaluation_results.json")


def test_metrics_summary_file():
    """Verify metrics summary CSV contains all 6 models and proper columns."""
    assert os.path.exists(METRICS_CSV), f"{METRICS_CSV} does not exist"
    df = pd.read_csv(METRICS_CSV)
    assert len(df) == 6, f"Expected 6 models in metrics summary, got {len(df)}"
    expected_cols = [
        "Model",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
        "ROC_AUC_OvR",
        "Latency_Per_Sample_ms",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column {col}"

    # Metric range sanity checks
    for metric in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"]:
        assert (df[metric] >= 0.0).all() and (df[metric] <= 1.0).all()


def test_confusion_matrices_shape_and_sum():
    """Verify confusion matrices in evaluation JSON have 3x3 shape and sum to test set size (885)."""
    assert os.path.exists(EVAL_JSON)
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    for model_name, data in eval_data.items():
        cm = np.array(data["confusion_matrix"])
        assert cm.shape == (3, 3), f"Confusion matrix for {model_name} should be 3x3, got {cm.shape}"
        assert cm.sum() == 885, f"Confusion matrix sum for {model_name} should be 885, got {cm.sum()}"

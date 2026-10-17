"""
Comprehensive unit tests for evaluation metrics, reporting artifacts, and error analysis.
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
PER_CLASS_CSV = os.path.join(REPORTS_DIR, "detailed_per_class_metrics.csv")
EVAL_JSON = os.path.join(REPORTS_DIR, "evaluation_results.json")


def test_metrics_summary_exists():
    """Verify metrics_summary.csv exists."""
    assert os.path.exists(METRICS_CSV)


def test_metrics_summary_row_count():
    """Verify metrics summary contains exactly 6 models."""
    df = pd.read_csv(METRICS_CSV)
    assert len(df) == 6, f"Expected 6 rows, got {len(df)}"


def test_metrics_summary_columns():
    """Verify all expected evaluation metrics exist as columns."""
    df = pd.read_csv(METRICS_CSV)
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
        assert col in df.columns, f"Missing {col} in metrics_summary"


def test_metrics_value_bounds():
    """Verify all percentage/ratio metrics are bounded in [0.0, 1.0]."""
    df = pd.read_csv(METRICS_CSV)
    for col in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all()


def test_no_nans_in_metrics_summary():
    """Verify no NaN values exist in metrics table."""
    df = pd.read_csv(METRICS_CSV)
    assert df.isnull().sum().sum() == 0, "Found NaNs in metrics_summary"


def test_cv_results_exists_and_rows():
    """Verify cv_results.csv exists and has 6 models."""
    assert os.path.exists(CV_CSV)
    cv_df = pd.read_csv(CV_CSV)
    assert len(cv_df) == 6
    assert "CV_Macro_F1_Mean" in cv_df.columns
    assert "CV_Macro_F1_Std" in cv_df.columns


def test_detailed_per_class_metrics_rows():
    """Verify detailed_per_class_metrics.csv has 18 rows (6 models * 3 classes)."""
    assert os.path.exists(PER_CLASS_CSV)
    df = pd.read_csv(PER_CLASS_CSV)
    assert len(df) == 18, f"Expected 18 rows, got {len(df)}"
    classes = set(df["Class"].unique())
    assert classes == {"Dropout", "Enrolled", "Graduate"}


def test_per_class_support_sums_to_test_size():
    """Verify test support across classes matches exactly 885."""
    df = pd.read_csv(PER_CLASS_CSV)
    for model in df["Model"].unique():
        sub = df[df["Model"] == model]
        assert sub["Support"].sum() == 885, f"Support sum for {model} is not 885"


def test_confusion_matrices_in_json():
    """Verify confusion matrices in evaluation_results.json are 3x3 and sum to 885."""
    assert os.path.exists(EVAL_JSON)
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    for model_name, info in data.items():
        cm = np.array(info["confusion_matrix"])
        assert cm.shape == (3, 3)
        assert cm.sum() == 885


def test_best_model_macro_f1_threshold():
    """Verify the top model achieves held-out test Macro F1 >= 0.70."""
    df = pd.read_csv(METRICS_CSV)
    top_f1 = df["Macro_F1"].max()
    assert top_f1 >= 0.70, f"Expected top Macro F1 >= 0.70, got {top_f1}"


def test_roc_auc_superior_to_chance():
    """Verify top models have ROC-AUC significantly above chance (> 0.85)."""
    df = pd.read_csv(METRICS_CSV)
    # Check Random Forest and Logistic Regression
    rf_auc = float(df[df["Model"] == "Random Forest"]["ROC_AUC_OvR"].values[0])
    lr_auc = float(df[df["Model"] == "Logistic Regression"]["ROC_AUC_OvR"].values[0])
    assert rf_auc > 0.85, f"Random Forest ROC-AUC {rf_auc} is below 0.85"
    assert lr_auc > 0.85, f"Logistic Regression ROC-AUC {lr_auc} is below 0.85"


def test_inference_latency_bounds():
    """Verify per-sample inference latency is positive and under 10 milliseconds."""
    df = pd.read_csv(METRICS_CSV)
    for _, row in df.iterrows():
        lat = row["Latency_Per_Sample_ms"]
        assert 0.0 < lat < 10.0, f"Latency {lat} ms is out of expected bounds"

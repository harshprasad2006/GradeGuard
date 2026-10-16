"""
Evaluation Module for GradeGuard.
Evaluates all trained classical ML models on the untouched held-out test set (885 samples).
Calculates overall and per-class metrics, confusion matrices, ROC-AUC (OvR), and inference latencies.
Exports results to CSV and JSON formats.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.abspath("."))
from src.data_processing import (
    TEST_DATA_PATH,
    TARGET_COLUMN,
    TARGET_ENCODING,
    INVERSE_TARGET_ENCODING,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_FILES = {
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Support Vector Machine": os.path.join(MODELS_DIR, "support_vector_machine.joblib"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "K-Nearest Neighbours": os.path.join(MODELS_DIR, "k_nearest_neighbours.joblib"),
    "Gaussian Naive Bayes": os.path.join(MODELS_DIR, "gaussian_naive_bayes.joblib"),
}

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
NUM_CLASSES = len(CLASS_NAMES)


def evaluate_all_models() -> Dict[str, Any]:
    """
    Evaluate all saved models on the held-out test set.
    """
    logger.info("Loading held-out test data from %s...", TEST_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].map(TARGET_ENCODING).values
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    summary_records = []
    per_class_records = []
    detailed_evaluation = {}

    for name, model_path in MODEL_FILES.items():
        if not os.path.exists(model_path):
            logger.warning("Model file %s not found. Skipping...", model_path)
            continue

        logger.info("Evaluating model: %s...", name)
        pipeline = joblib.load(model_path)

        # Inference timing
        t0 = time.time()
        y_pred = pipeline.predict(X_test)
        inference_time_ms = (time.time() - t0) * 1000.0 / len(X_test)

        # Probabilities for ROC-AUC
        y_proba = None
        roc_auc_ovr = np.nan
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
                roc_auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
            except Exception as e:
                logger.warning("Failed to compute ROC-AUC for %s: %e", name, e)
        elif hasattr(pipeline, "decision_function"):
            try:
                dec = pipeline.decision_function(X_test)
                # Softmax decision function
                exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                y_proba = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
                roc_auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
            except Exception as e:
                logger.warning("Decision function ROC-AUC failed for %s: %e", name, e)

        # Overall metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        summary_records.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Macro_Precision": round(float(prec_macro), 4),
            "Macro_Recall": round(float(rec_macro), 4),
            "Macro_F1": round(float(f1_macro), 4),
            "Weighted_F1": round(float(f1_weighted), 4),
            "ROC_AUC_OvR": round(float(roc_auc_ovr), 4) if not np.isnan(roc_auc_ovr) else "N/A",
            "Latency_Per_Sample_ms": round(float(inference_time_ms), 3),
        })

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            per_class_records.append({
                "Model": name,
                "Class": cls_name,
                "Precision": round(float(prec_per_class[cls_idx]), 4),
                "Recall": round(float(rec_per_class[cls_idx]), 4),
                "F1_Score": round(float(f1_per_class[cls_idx]), 4),
                "Support": int(np.sum(y_test == cls_idx)),
            })

        detailed_evaluation[name] = {
            "overall": {
                "accuracy": float(acc),
                "macro_precision": float(prec_macro),
                "macro_recall": float(rec_macro),
                "macro_f1": float(f1_macro),
                "weighted_f1": float(f1_weighted),
                "roc_auc_ovr": float(roc_auc_ovr) if not np.isnan(roc_auc_ovr) else None,
                "latency_per_sample_ms": float(inference_time_ms),
            },
            "per_class": {
                cls_name: {
                    "precision": float(prec_per_class[i]),
                    "recall": float(rec_per_class[i]),
                    "f1_score": float(f1_per_class[i]),
                    "support": int(np.sum(y_test == i)),
                }
                for i, cls_name in enumerate(CLASS_NAMES)
            },
            "confusion_matrix": cm.tolist(),
        }

    # Save summary table
    summary_df = pd.DataFrame(summary_records).sort_values(by="Macro_F1", ascending=False)
    summary_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved metrics summary to %s", summary_path)

    # Save per-class table
    per_class_df = pd.DataFrame(per_class_records)
    per_class_path = os.path.join(REPORTS_DIR, "detailed_per_class_metrics.csv")
    per_class_df.to_csv(per_class_path, index=False)
    logger.info("Saved detailed per-class metrics to %s", per_class_path)

    # Save detailed JSON
    json_path = os.path.join(REPORTS_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_evaluation, f, indent=2)
    logger.info("Saved evaluation JSON to %s", json_path)

    return detailed_evaluation


if __name__ == "__main__":
    evaluate_all_models()

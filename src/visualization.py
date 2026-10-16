"""
Visualization Module for GradeGuard.
Generates publication-quality charts and plots for the report and dashboard:
- Class distribution across dataset splits
- Model metric comparisons
- Confusion matrices
- Multiclass ROC curves (One-vs-Rest)
- Feature importances
- Precision-Recall curves
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.abspath("."))
from src.data_processing import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    TARGET_ENCODING,
    INVERSE_TARGET_ENCODING,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
)
from src.evaluate import MODEL_FILES, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Aesthetic palette (Navy/Teal/Slate)
COLOR_PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "slate": "#5C768D",
    "amber": "#D97706",
    "coral": "#DC2626",
    "light_gray": "#F3F4F6",
    "grid_gray": "#E5E7EB",
}

plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def plot_class_distribution():
    """Generate class distribution chart comparing Full, Train, and Test sets."""
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    categories = CLASS_NAMES
    raw_counts = [df_raw[TARGET_COLUMN].value_counts().get(c, 0) for c in categories]
    train_counts = [df_train[TARGET_COLUMN].value_counts().get(c, 0) for c in categories]
    test_counts = [df_test[TARGET_COLUMN].value_counts().get(c, 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width, raw_counts, width, label="Full Dataset (N=4,424)", color="#1B365D")
    rects2 = ax.bar(x, train_counts, width, label="Training Set (80%, N=3,539)", color="#008080")
    rects3 = ax.bar(x + width, test_counts, width, label="Held-Out Test Set (20%, N=885)", color="#5C768D")

    ax.set_ylabel("Number of Students", fontsize=11, fontweight="bold")
    ax.set_title("Student Outcome Class Distribution Across Stratified Splits", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add count labels on bars
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f"{height}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_model_metric_comparison():
    """Generate grouped bar chart comparing all models on held-out test metrics."""
    metrics_csv = os.path.join("reports", "metrics_summary.csv")
    df = pd.read_csv(metrics_csv)
    
    # Exclude Gaussian Naive Bayes from primary competitive plot for visual scaling or show all
    # Showing all with clear labeling
    models = df["Model"].tolist()
    metrics = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"]
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]

    x = np.arange(len(models))
    width = 0.15
    colors = ["#1B365D", "#008080", "#5C768D", "#D97706", "#2563EB"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
        vals = df[m].tolist()
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=c)

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Classical ML Models: Held-Out Test Evaluation Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "model_metric_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_confusion_matrices():
    """Plot confusion matrices for all 6 models in a 2x3 grid."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].map(TARGET_ENCODING).values

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()

    for idx, (name, path) in enumerate(MODEL_FILES.items()):
        ax = axes[idx]
        if not os.path.exists(path):
            ax.set_title(f"{name} (Not Found)")
            continue

        model = joblib.load(path)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold", pad=8)
        
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_ylabel("True Outcome", fontsize=9)
        ax.set_xlabel("Predicted Outcome", fontsize=9)

        # Annotate cells with count and percentage
        thresh = cm_norm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                        ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_roc_curves():
    """Plot multiclass One-vs-Rest ROC curves for Random Forest, Logistic Regression, and SVM."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].map(TARGET_ENCODING).values
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    selected_models = ["Random Forest", "Logistic Regression", "Support Vector Machine"]

    for ax_idx, model_name in enumerate(selected_models):
        ax = axes[ax_idx]
        path = MODEL_FILES[model_name]
        pipeline = joblib.load(path)
        y_proba = pipeline.predict_proba(X_test)

        colors = ["#DC2626", "#D97706", "#059669"]
        for i, color in zip(range(len(CLASS_NAMES)), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})")

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, color="#1B365D", linestyle=":", lw=2,
                label=f"Micro-avg (AUC = {auc_micro:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(f"{model_name}\nOne-vs-Rest ROC", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_feature_importance():
    """Extract and plot top feature importances from Random Forest pipeline."""
    rf_path = MODEL_FILES["Random Forest"]
    rf_pipeline = joblib.load(rf_path)
    
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_model = rf_pipeline.named_steps["model"]

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(NOMINAL_CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + BINARY_FEATURES + cat_feature_names

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals, align="center", color="#008080", edgecolor="#1B365D", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (Gini Feature Importance)", fontsize=10, fontweight="bold")
    ax.set_title("Random Forest: Top 20 Most Predictive Features", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)


def plot_pr_curves():
    """Generate multiclass Precision-Recall curves."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].map(TARGET_ENCODING).values
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    rf_path = MODEL_FILES["Random Forest"]
    rf_pipeline = joblib.load(rf_path)
    y_proba = rf_pipeline.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    colors = ["#DC2626", "#D97706", "#059669"]

    for i, color in zip(range(len(CLASS_NAMES)), colors):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{CLASS_NAMES[i]} (Avg Prec = {ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax.set_title("Precision-Recall Curves (Random Forest)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "pr_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)


def generate_all_visualizations():
    logger.info("Generating publication figures in %s...", FIGURES_DIR)
    plot_class_distribution()
    plot_model_metric_comparison()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_feature_importance()
    plot_pr_curves()
    logger.info("All figures generated successfully.")


if __name__ == "__main__":
    generate_all_visualizations()

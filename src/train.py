"""
Model Training and Hyperparameter Tuning Module for GradeGuard.
Trains 6 classical ML classifiers using StratifiedKFold cross-validation
and scikit-learn Pipelines. Evaluates class weighting, selects the best model,
and saves all artifacts with joblib.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import sys
sys.path.insert(0, os.path.abspath("."))

from src.data_processing import (
    get_preprocessor,
    TRAIN_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TARGET_ENCODING,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def define_models_and_grids() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Define 6 classical ML classifiers and computationally reasonable hyperparameter grids.
    Includes class weighting exploration for algorithms that support it.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["lbfgs"],
            },
        ),
        "K-Nearest Neighbours": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [5, 9, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["minkowski", "manhattan"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 15, None],
                "model__min_samples_split": [2, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1.0, 5.0],
                "model__kernel": ["rbf", "linear"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gaussian Naive Bayes": (
            GaussianNB(),
            {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        ),
    }


def train_and_tune_all() -> Dict[str, Any]:
    """
    Execute 5-fold Stratified CV hyperparameter tuning across all models.
    Selects the best model based strictly on validation Macro F1 score.
    """
    logger.info("Loading training data from %s...", TRAIN_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].map(TARGET_ENCODING)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_defs = define_models_and_grids()

    results_records = []
    fitted_pipelines = {}
    best_params_dict = {}

    logger.info("Starting hyperparameter tuning for %d models...", len(model_defs))

    for name, (estimator, param_grid) in model_defs.items():
        logger.info("-" * 50)
        logger.info("Tuning model: %s", name)
        
        # Build leak-free pipeline with preprocessor
        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("model", estimator),
        ])

        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=["f1_macro", "accuracy", "f1_weighted"],
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        grid_search.fit(X_train, y_train)
        fit_time = time.time() - start_time

        best_idx = grid_search.best_index_
        best_macro_f1 = grid_search.cv_results_["mean_test_f1_macro"][best_idx]
        std_macro_f1 = grid_search.cv_results_["std_test_f1_macro"][best_idx]
        best_accuracy = grid_search.cv_results_["mean_test_accuracy"][best_idx]
        std_accuracy = grid_search.cv_results_["std_test_accuracy"][best_idx]
        best_weighted_f1 = grid_search.cv_results_["mean_test_f1_weighted"][best_idx]
        std_weighted_f1 = grid_search.cv_results_["std_test_f1_weighted"][best_idx]

        logger.info(
            "%s Best CV Macro F1: %.4f (+/- %.4f) | CV Accuracy: %.4f (+/- %.4f) | Fit Time: %.2fs",
            name,
            best_macro_f1,
            std_macro_f1,
            best_accuracy,
            std_accuracy,
            fit_time,
        )
        logger.info("Best Parameters: %s", grid_search.best_params_)

        # Clean filename slug
        slug = name.lower().replace(" ", "_").replace("-", "_")
        model_filename = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(grid_search.best_estimator_, model_filename)
        logger.info("Saved pipeline to %s", model_filename)

        fitted_pipelines[name] = grid_search.best_estimator_
        best_params_dict[name] = grid_search.best_params_

        results_records.append({
            "Model": name,
            "CV_Macro_F1_Mean": round(float(best_macro_f1), 4),
            "CV_Macro_F1_Std": round(float(std_macro_f1), 4),
            "CV_Accuracy_Mean": round(float(best_accuracy), 4),
            "CV_Accuracy_Std": round(float(std_accuracy), 4),
            "CV_Weighted_F1_Mean": round(float(best_weighted_f1), 4),
            "CV_Weighted_F1_Std": round(float(std_weighted_f1), 4),
            "Fit_Time_Seconds": round(float(fit_time), 2),
            "Best_Params": str(grid_search.best_params_),
            "Saved_Path": model_filename,
        })

    cv_df = pd.DataFrame(results_records).sort_values(by="CV_Macro_F1_Mean", ascending=False)
    cv_csv_path = os.path.join(REPORTS_DIR, "cv_results.csv")
    cv_df.to_csv(cv_csv_path, index=False)
    logger.info("Saved CV summary table to %s", cv_csv_path)

    # Select the genuine best model based on validation macro F1
    best_row = cv_df.iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    logger.info(
        "BEST MODEL SELECTED: '%s' with CV Macro F1 = %.4f. Saved to %s",
        best_model_name,
        best_row["CV_Macro_F1_Mean"],
        best_model_path,
    )

    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "CV Macro F1 (Stratified 5-Fold)",
        "best_cv_macro_f1": float(best_row["CV_Macro_F1_Mean"]),
        "best_cv_accuracy": float(best_row["CV_Accuracy_Mean"]),
        "best_params": best_params_dict[best_model_name],
        "all_models": list(model_defs.keys()),
        "cv_results": results_records,
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model training metadata to %s", metadata_path)

    return metadata


if __name__ == "__main__":
    train_and_tune_all()

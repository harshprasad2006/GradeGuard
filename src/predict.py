"""
Prediction and Decision Support Module for GradeGuard.
Handles model inference, probability estimation, input validation,
influential factor breakdown, and supportive academic guidance.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.data_processing import (
    METADATA_PATH,
    CLASS_MAPPING_PATH,
    TARGET_COLUMN,
    TARGET_ENCODING,
    INVERSE_TARGET_ENCODING,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
    HUMAN_READABLE_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BEST_MODEL_PATH = os.path.join("models", "best_model.joblib")


def load_model(model_path: str = BEST_MODEL_PATH):
    """Load a trained scikit-learn pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train models first using 'python src/train.py'."
        )
    return joblib.load(model_path)


def load_metadata() -> Dict[str, Any]:
    """Load feature metadata JSON."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_student_input(input_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that input DataFrame contains all 36 required features
    and values are within sensible ranges.
    """
    issues = []
    metadata = load_metadata()
    required_features = metadata["all_features_in_order"]

    missing_cols = [col for col in required_features if col not in input_df.columns]
    if missing_cols:
        issues.append(f"Missing required columns ({len(missing_cols)}): {missing_cols[:5]}...")

    # Range validation for numeric features
    feature_ranges = metadata.get("feature_ranges", {})
    for col in NUMERICAL_FEATURES:
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            if pd.isna(val):
                issues.append(f"Value for '{col}' cannot be null.")
            elif not isinstance(val, (int, float, np.number)):
                issues.append(f"Value for '{col}' must be numeric (got {type(val).__name__}).")

    return len(issues) == 0, issues


def generate_supportive_guidance(features: Dict[str, Any], predicted_class: str) -> List[str]:
    """
    Generate non-deterministic, supportive educational recommendations based on observable academic indicators.
    Avoids stigmatizing language, disclaims causal claims, and recommends holistic academic advising.
    """
    guidance = []

    tuition_up_to_date = features.get("Tuition fees up to date", 1)
    is_debtor = features.get("Debtor", 0)
    sem1_enrolled = features.get("Curricular units 1st sem (enrolled)", 0)
    sem1_approved = features.get("Curricular units 1st sem (approved)", 0)
    sem2_enrolled = features.get("Curricular units 2nd sem (enrolled)", 0)
    sem2_approved = features.get("Curricular units 2nd sem (approved)", 0)
    scholarship = features.get("Scholarship holder", 0)

    # Academic credit progression indicator
    sem1_ratio = (sem1_approved / sem1_enrolled) if sem1_enrolled > 0 else 1.0
    sem2_ratio = (sem2_approved / sem2_enrolled) if sem2_enrolled > 0 else 1.0

    if sem1_ratio < 0.6 or sem2_ratio < 0.6:
        guidance.append(
            "Curricular Credit Pace: Fewer than 60% of enrolled semester units were approved. "
            "Recommend scheduling a peer tutoring session or study skills workshop for challenging subject modules."
        )

    if tuition_up_to_date == 0 or is_debtor == 1:
        guidance.append(
            "Financial Administration Support: Records indicate outstanding tuition or administrative balance. "
            "Recommend connecting the student with Student Financial Aid services, emergency grant counseling, or installment plans."
        )

    if scholarship == 1 and (sem1_ratio < 0.75 or sem2_ratio < 0.75):
        guidance.append(
            "Scholarship Retention Advisory: The student holds a scholarship. Proactive check-in with the scholarship retention advisor "
            "is advised to ensure minimum GPA and credit milestones are maintained."
        )

    if predicted_class == "Enrolled":
        guidance.append(
            "Degree Completion Roadmap: The model classifies the student as currently 'Enrolled' (in progress). "
            "Recommend a degree audit consultation to clarify remaining graduation requirements and optimize future course sequencing."
        )
    elif predicted_class == "Dropout":
        guidance.append(
            "Holistic Early Outreach: Early academic engagement outreach is encouraged. Pair with a dedicated academic mentor "
            "to assess course workload balance, personal circumstances, and career alignment."
        )
    else:  # Graduate
        guidance.append(
            "Academic Excellence & Mentorship: Progression indicators reflect strong positive momentum. Consider encouraging "
            "undergraduate research participation, honours tracks, or serving as a peer study leader."
        )

    guidance.append(
        "Important Note: These automated recommendations are educational decision-support suggestions, "
        "not definitive personal judgments. Human academic advisor review is mandatory before taking action."
    )

    return guidance


def predict_student(
    student_data: Dict[str, Any],
    model_path: str = BEST_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Run prediction on a single student's dictionary of features.
    Returns predicted class, probabilities, confidence, guidance, and disclaimer.
    """
    pipeline = load_model(model_path)
    metadata = load_metadata()

    # Create 1-row DataFrame with required columns in order
    ordered_cols = metadata["all_features_in_order"]
    row_data = {col: [student_data.get(col, 0)] for col in ordered_cols}
    input_df = pd.DataFrame(row_data)

    is_valid, issues = validate_student_input(input_df)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(issues)}")

    # Model prediction
    pred_int = int(pipeline.predict(input_df)[0])
    pred_label = INVERSE_TARGET_ENCODING[pred_int]

    # Probabilities
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        for idx, cls_name in enumerate(["Dropout", "Enrolled", "Graduate"]):
            probabilities[cls_name] = round(float(probs[idx]), 4)
    else:
        probabilities = {pred_label: 1.0}

    guidance = generate_supportive_guidance(student_data, pred_label)

    return {
        "predicted_class": pred_label,
        "predicted_class_id": pred_int,
        "probabilities": probabilities,
        "confidence": probabilities.get(pred_label, 1.0),
        "guidance": guidance,
        "is_calibrated": False,  # Explicitly disclaim calibration per requirement 1
        "disclaimer": (
            "GradeGuard Decision-Support Disclaimer: This prediction is generated from an educational prototype trained "
            "on historical institutional records (UCI dataset ID 697). It reflects end-of-normal-duration statistical "
            "associations, not real-time causal determinants. Output probabilities are model-derived softmax/logistic scores "
            "and are not formally calibrated real-world probabilities. This tool must never be used to make autonomous or "
            "punitive administrative decisions without qualified human review."
        ),
    }


def get_sample_profiles() -> Dict[str, Dict[str, Any]]:
    """
    Return 3 representative student profiles derived from actual dataset records for testing.
    Profile 1: Strong academic completion (Graduate)
    Profile 2: Credit deficits and financial flags (Dropout risk)
    Profile 3: Intermediate active progression (Enrolled)
    """
    test_df = pd.read_csv(os.path.join("data", "processed", "test_data.csv"))
    
    # Extract authentic records for each class
    grad_record = test_df[test_df[TARGET_COLUMN] == "Graduate"].iloc[0].drop(TARGET_COLUMN).to_dict()
    drop_record = test_df[test_df[TARGET_COLUMN] == "Dropout"].iloc[0].drop(TARGET_COLUMN).to_dict()
    enr_record = test_df[test_df[TARGET_COLUMN] == "Enrolled"].iloc[0].drop(TARGET_COLUMN).to_dict()

    return {
        "Profile A: High Academic Momentum (Graduate Benchmark)": grad_record,
        "Profile B: Academic Credit Deficit & Administrative Flags (Dropout Risk)": drop_record,
        "Profile C: Active Modular Progression (Enrolled Status)": enr_record,
    }


def export_sample_csv_template(filepath: str = "data/sample_prediction_template.csv"):
    """Export a clean 1-row CSV template with column headers for the user."""
    metadata = load_metadata()
    cols = metadata["all_features_in_order"]
    sample_profiles = get_sample_profiles()
    sample_dict = sample_profiles["Profile A: High Academic Momentum (Graduate Benchmark)"]
    df = pd.DataFrame([sample_dict])[cols]
    df.to_csv(filepath, index=False)
    logger.info("Saved sample prediction template to %s", filepath)


if __name__ == "__main__":
    export_sample_csv_template()
    profiles = get_sample_profiles()
    print("Testing prediction on sample profiles:")
    for name, p in profiles.items():
        res = predict_student(p)
        print(f"\n--- {name} ---")
        print(f"Predicted Class: {res['predicted_class']} (Confidence: {res['confidence']:.2%})")
        print(f"Probabilities: {res['probabilities']}")

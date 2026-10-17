"""
Comprehensive unit tests for inference engine, input validation, and decision support guidance.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.predict import (
    predict_student,
    get_sample_profiles,
    validate_student_input,
    generate_supportive_guidance,
    load_metadata,
)

METADATA = load_metadata()
ALL_FEATURES = METADATA["all_features_in_order"]


def test_sample_profiles_count_and_keys():
    """Verify get_sample_profiles returns exactly 3 benchmark profiles."""
    profiles = get_sample_profiles()
    assert len(profiles) == 3
    for name, p in profiles.items():
        assert len(p) == 36, f"Profile {name} should have 36 features, got {len(p)}"


def test_predict_output_structure():
    """Verify predict_student returns all required keys."""
    profiles = get_sample_profiles()
    first_profile = list(profiles.values())[0]
    res = predict_student(first_profile)
    
    assert "predicted_class" in res
    assert "predicted_class_id" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert "guidance" in res
    assert "disclaimer" in res
    assert "is_calibrated" in res


def test_predicted_class_valid_labels():
    """Verify predicted_class is strictly one of the 3 target outcomes."""
    profiles = get_sample_profiles()
    for name, p in profiles.items():
        res = predict_student(p)
        assert res["predicted_class"] in ["Dropout", "Enrolled", "Graduate"]
        assert res["predicted_class_id"] in [0, 1, 2]


def test_probabilities_sum_to_one():
    """Verify probabilities dictionary contains all 3 classes and sums to 1.0."""
    profiles = get_sample_profiles()
    for name, p in profiles.items():
        res = predict_student(p)
        probs = res["probabilities"]
        assert set(probs.keys()) == {"Dropout", "Enrolled", "Graduate"}
        total_p = sum(probs.values())
        assert abs(total_p - 1.0) < 0.02, f"Probabilities sum to {total_p}"


def test_confidence_matches_predicted_class():
    """Verify confidence score equals the probability of the predicted class."""
    profiles = get_sample_profiles()
    for name, p in profiles.items():
        res = predict_student(p)
        pred = res["predicted_class"]
        conf = res["confidence"]
        assert conf == res["probabilities"][pred]


def test_case_high_performer_predicts_graduate():
    """Verify student with 6/6 units approved and high GPA predicts Graduate."""
    student = {feat: 0 for feat in ALL_FEATURES}
    student.update({
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 15.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 15.5,
        "Tuition fees up to date": 1,
        "Debtor": 0,
        "Admission grade": 140.0,
        "Age at enrollment": 20,
    })
    res = predict_student(student)
    assert res["predicted_class"] == "Graduate"
    assert res["confidence"] > 0.60


def test_case_severe_failure_predicts_dropout():
    """Verify student with 0 units approved in both semesters predicts Dropout."""
    student = {feat: 0 for feat in ALL_FEATURES}
    student.update({
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 0,
        "Curricular units 1st sem (grade)": 0.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 0,
        "Curricular units 2nd sem (grade)": 0.0,
        "Tuition fees up to date": 0,
        "Debtor": 1,
        "Age at enrollment": 24,
    })
    res = predict_student(student)
    assert res["predicted_class"] == "Dropout"
    assert res["confidence"] > 0.60


def test_case_sem2_slump_predicts_dropout():
    """Verify student who passes sem 1 but fails sem 2 is flagged as Dropout risk."""
    student = {feat: 0 for feat in ALL_FEATURES}
    student.update({
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 13.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 0,
        "Curricular units 2nd sem (grade)": 0.0,
        "Tuition fees up to date": 1,
        "Debtor": 0,
    })
    res = predict_student(student)
    assert res["predicted_class"] == "Dropout"


def test_input_validation_detects_missing_columns():
    """Verify validation fails if required columns are omitted."""
    incomplete_df = pd.DataFrame([{"Gender": 1, "Admission grade": 120}])
    is_valid, issues = validate_student_input(incomplete_df)
    assert not is_valid
    assert any("Missing required columns" in issue for issue in issues)


def test_input_validation_detects_nulls():
    """Verify validation detects null values in numeric features."""
    sample = {feat: 1.0 for feat in ALL_FEATURES}
    sample["Admission grade"] = np.nan
    df = pd.DataFrame([sample])
    is_valid, issues = validate_student_input(df)
    assert not is_valid
    assert any("cannot be null" in issue for issue in issues)


def test_guidance_credit_pace_recommendation():
    """Verify guidance recommends tutoring when credit approval rate is below 60%."""
    feats = {
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 1,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 1,
        "Tuition fees up to date": 1,
        "Debtor": 0,
    }
    guidance = generate_supportive_guidance(feats, "Dropout")
    assert any("Curricular Credit Pace" in g or "tutoring" in g for g in guidance)


def test_guidance_financial_aid_recommendation():
    """Verify guidance recommends financial counseling for debtors/unpaid tuition."""
    feats = {
        "Tuition fees up to date": 0,
        "Debtor": 1,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 6,
    }
    guidance = generate_supportive_guidance(feats, "Dropout")
    assert any("Financial Administration Support" in g or "emergency grant" in g for g in guidance)


def test_guidance_scholarship_advisory():
    """Verify scholarship retention warning when scholarship holder has low credits."""
    feats = {
        "Scholarship holder": 1,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 2,
    }
    guidance = generate_supportive_guidance(feats, "Dropout")
    assert any("Scholarship Retention Advisory" in g for g in guidance)


def test_prediction_template_file_structure():
    """Verify sample prediction template exists and has exactly 36 columns."""
    tmpl_path = "data/sample_prediction_template.csv"
    assert os.path.exists(tmpl_path)
    df = pd.read_csv(tmpl_path)
    assert df.shape[1] == 36
    # Should predict cleanly without error
    record = df.iloc[0].to_dict()
    res = predict_student(record)
    assert res["predicted_class"] in ["Dropout", "Enrolled", "Graduate"]

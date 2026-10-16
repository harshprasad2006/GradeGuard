"""
Unit tests for inference, input validation, and decision support guidance.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.predict import (
    predict_student,
    get_sample_profiles,
    validate_student_input,
    generate_supportive_guidance,
)


def test_predict_sample_profiles():
    """Verify prediction produces valid class, probabilities, and guidance for all sample profiles."""
    profiles = get_sample_profiles()
    assert len(profiles) == 3

    for name, p in profiles.items():
        res = predict_student(p)
        assert res["predicted_class"] in ["Dropout", "Enrolled", "Graduate"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert set(res["probabilities"].keys()) == {"Dropout", "Enrolled", "Graduate"}
        prob_sum = sum(res["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.02, f"Probabilities do not sum to 1: {prob_sum}"
        assert len(res["guidance"]) >= 2
        assert "disclaimer" in res


def test_input_validation_catches_missing():
    """Verify validate_student_input flags missing columns."""
    incomplete_df = pd.DataFrame([{"Gender": 1, "Age at enrollment": 20}])
    is_valid, issues = validate_student_input(incomplete_df)
    assert not is_valid
    assert len(issues) > 0


def test_supportive_guidance_generation():
    """Verify non-empty supportive recommendations for students with credit/fee indicators."""
    test_features = {
        "Tuition fees up to date": 0,
        "Debtor": 1,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 1,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 1,
        "Scholarship holder": 1,
    }
    guidance = generate_supportive_guidance(test_features, "Dropout")
    assert any("Financial" in g or "Tuition" in g for g in guidance)
    assert any("Credit" in g or "Pace" in g for g in guidance)
    assert any("Scholarship" in g for g in guidance)

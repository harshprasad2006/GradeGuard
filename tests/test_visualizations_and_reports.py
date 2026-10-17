"""
Unit tests for visual figures, 10-page reports, and viva preparation documentation.
"""

import os
import re
import pytest

FIGURES_DIR = os.path.join("reports", "figures")
DOCX_REPORT_PATH = os.path.join("reports", "GradeGuard_Report.docx")
PDF_REPORT_PATH = os.path.join("reports", "GradeGuard_Report.pdf")
VIVA_PATH = os.path.join("reports", "Viva_Questions.md")


def test_all_six_figures_exist():
    """Verify all 6 publication figures exist in reports/figures/."""
    expected_figures = [
        "class_distribution.png",
        "model_metric_comparison.png",
        "confusion_matrices.png",
        "roc_curves.png",
        "feature_importance.png",
        "pr_curves.png",
    ]
    for fig_name in expected_figures:
        path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(path), f"Figure {fig_name} is missing"


def test_figures_file_sizes_healthy():
    """Verify figures are not empty or corrupted (file size > 50 KB)."""
    expected_figures = [
        "class_distribution.png",
        "model_metric_comparison.png",
        "confusion_matrices.png",
        "roc_curves.png",
        "feature_importance.png",
        "pr_curves.png",
    ]
    for fig_name in expected_figures:
        path = os.path.join(FIGURES_DIR, fig_name)
        size_kb = os.path.getsize(path) / 1024.0
        assert size_kb > 50.0, f"Figure {fig_name} is too small ({size_kb:.1f} KB)"


def test_docx_report_exists_and_size():
    """Verify GradeGuard_Report.docx exists and is > 100 KB."""
    assert os.path.exists(DOCX_REPORT_PATH)
    size_kb = os.path.getsize(DOCX_REPORT_PATH) / 1024.0
    assert size_kb > 100.0, f"DOCX report size {size_kb:.1f} KB is unexpectedly small"


def test_pdf_report_exists_and_size():
    """Verify GradeGuard_Report.pdf exists and is > 100 KB."""
    assert os.path.exists(PDF_REPORT_PATH)
    size_kb = os.path.getsize(PDF_REPORT_PATH) / 1024.0
    assert size_kb > 100.0, f"PDF report size {size_kb:.1f} KB is unexpectedly small"


def test_pdf_report_exact_ten_pages():
    """Verify GradeGuard_Report.pdf has exactly 10 pages."""
    with open(PDF_REPORT_PATH, "rb") as f:
        content = f.read()
    page_matches = re.findall(rb'/Type\s*/Page\b', content)
    assert len(page_matches) == 10, f"Expected exactly 10 pages in PDF report, found {len(page_matches)}"


def test_viva_questions_contains_minimum_count():
    """Verify Viva_Questions.md contains at least 25 questions and answers."""
    assert os.path.exists(VIVA_PATH)
    with open(VIVA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # Check for question headers like '#### Q1:', '#### Q25:'
    question_headers = re.findall(r'####\s*Q\d+:', text)
    assert len(question_headers) >= 25, f"Expected >= 25 viva questions, found {len(question_headers)}"
    assert "60-Second Executive Pitch" in text
    assert "Walkthrough of How One Prediction is Generated" in text

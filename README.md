# 🎓 GradeGuard: Student Academic Success and Dropout Risk Prediction

> **A reproducible, classical machine learning decision-support system predicting student academic outcomes (Dropout, Enrolled, Graduate) using the official UCI Machine Learning Repository Dataset ID 697.**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/UCI%20Repository-ID%20697-green.svg)](https://doi.org/10.24432/C5MC89)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/pytest-14%20passed-brightgreen.svg)]()

---

## 📌 Executive Summary

GradeGuard is an assessment-ready machine learning mini-project designed to evaluate higher education student progression and retention. Using the official, peer-reviewed dataset *Predict Students' Dropout and Academic Success* (UCI ID 697, DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)), the system predicts one of three mutually exclusive academic outcomes:

1. **Dropout:** Students who terminate enrollment prior to completion.
2. **Enrolled:** Students actively progressing through modular coursework who remain in good standing but have not completed degree requirements within the nominal duration.
3. **Graduate:** Students who successfully complete all graduation criteria.

### ⚠️ Critical Methodological Bounds
- **Temporal Horizon:** The dataset includes curricular units credited, enrolled, evaluated, and approved from both the **1st and 2nd semesters**. This is an **end-of-normal-duration (or post-first-year) evaluation**, **not** a day-one admissions early warning tool. Day-one early warning models must exclude post-matriculation academic performance.
- **Probability Calibration:** Model probabilities represent softmax confidence scores or tree leaf frequencies. They have **not** undergone empirical calibration and must not be interpreted as actuarial risk percentages.
- **Decision Support Prototype:** GradeGuard is an educational prototype designed to assist human academic advisors. It is strictly prohibited from making automated punitive decisions (e.g., aid revocation, course gating).

---

## 🔬 Literature Review (Verified 2023–2026 DOIs)

| Study & DOI | Dataset & Scope | Evaluated Models | Key Findings | Relevance to GradeGuard |
| :--- | :--- | :--- | :--- | :--- |
| **Villar & de Andrade (2024)**<br>[10.1007/s44163-023-00079-z](https://doi.org/10.1007/s44163-023-00079-z) | UCI 697 (Portugal)<br>$N = 4,424$ | LightGBM, CatBoost, XGBoost, RF, SVM | Optuna-tuned gradient boosting outperformed classical trees; SMOTE mitigated class imbalance. | Benchmarks performance on UCI 697; motivated our leak-free class weighting comparison. |
| **Vaarma & Li (2024)**<br>[10.1016/j.techsoc.2024.102474](https://doi.org/10.1016/j.techsoc.2024.102474) | Finnish Higher Ed<br>LMS & Transcripts | Logistic Regression, Random Forest, XGBoost | Accumulated credits and failed courses were primary retention predictors; monthly LMS tracking added longitudinal granularity. | Affirms that curricular credit progress dominates pre-entry demographics in explanatory power. |
| **Christou et al. (2023)**<br>[10.1016/j.eswa.2023.120079](https://doi.org/10.1016/j.eswa.2023.120079) | Greek Higher Ed<br>Undergrad Cohort | SVM, Neural Networks, Decision Trees, KNN | Early drop-out detection accuracy improved substantially when 1st-year examination outcomes were incorporated. | Reinforces our emphasis on distinguishing post-first-year prediction from day-one early warning. |
| **Krüger et al. (2023)**<br>[10.1016/j.eswa.2023.120933](https://doi.org/10.1016/j.eswa.2023.120933) | Brazilian Cohort<br>19 Schools (K-12/Sec) | Random Forest, Extra Trees, XGBoost + SHAP | Emphasized Explainable AI (XAI) to uncover root causes behind dropout across distinct educational stages and mitigate bias. | Guided GradeGuard's feature importance analysis and non-deterministic supportive guidance module. |
| **Cho, Yu & Kim (2023)**<br>[10.3390/app132112004](https://doi.org/10.3390/app132112004) | South Korean Univ<br>Cohort Registry | Logistic Regression, Random Forest, LightGBM | Cost-sensitive learning and class weighting achieved higher recall on dropouts than post-hoc probability threshold tuning. | Directly justified our selection of balanced class weighting within cross-validation folds. |
| **Realinho et al. (2021)**<br>[10.3390/data6110110](https://doi.org/10.3390/data6110110) | Politécnico Portalegre<br>UCI ID 697 ($N=4,424$) | Random Forest, Gradient Boosting, SVM | Established the foundational 36-feature dataset; 2nd semester approved units was identified as single strongest outcome predictor. | Official dataset origin and empirical baseline for GradeGuard. |

---

## 📊 Empirical Results & Benchmark Summary

All models were evaluated on the **untouched held-out test set ($N=885$, 20% Stratified Split)** after 5-fold Stratified Cross-Validation on the training cohort ($N=3,539$).

### Held-Out Test Set Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC (OvR) | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Selected Best)** | **0.7593** | **0.7067** | **0.7011** | **0.7028** | **0.7597** | 0.8784 | 0.054 ms |
| **Logistic Regression** | 0.7367 | 0.7091 | 0.7131 | 0.7001 | 0.7499 | **0.8849** | **0.010 ms** |
| **Support Vector Machine (SVM)** | 0.7277 | 0.6950 | 0.6955 | 0.6874 | 0.7382 | 0.8837 | 0.240 ms |
| **Decision Tree** | 0.6768 | 0.6763 | 0.6639 | 0.6480 | 0.6977 | 0.8297 | 0.009 ms |
| **K-Nearest Neighbours (KNN)** | 0.6938 | 0.6294 | 0.5978 | 0.6050 | 0.6774 | 0.7979 | 0.146 ms |
| **Gaussian Naive Bayes** | 0.2576 | 0.4822 | 0.4053 | 0.2350 | 0.1840 | 0.6911 | 0.014 ms |

### 5-Fold Stratified Cross-Validation (Training Set, $N=3,539$)

| Model | CV Macro F1 (Mean ± Std) | CV Accuracy (Mean ± Std) | CV Weighted F1 (Mean ± Std) | Fit Time | Selected Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **0.7263 ± 0.0194** | **0.7773 ± 0.0150** | **0.7758 ± 0.0156** | 30.3 s | `class_weight='balanced', max_depth=None, n_estimators=100` |
| **Logistic Regression** | 0.7174 ± 0.0182 | 0.7578 ± 0.0157 | 0.7659 ± 0.0140 | 7.36 s | `C=0.1, class_weight='balanced', solver='lbfgs'` |
| **SVM (RBF Kernel)** | 0.7134 ± 0.0168 | 0.7536 ± 0.0136 | 0.7613 ± 0.0143 | 141.9 s | `C=1.0, class_weight='balanced', kernel='rbf'` |
| **Decision Tree** | 0.6715 ± 0.0125 | 0.7058 ± 0.0216 | 0.7213 ± 0.0160 | 4.42 s | `class_weight='balanced', max_depth=5, min_samples_split=2` |
| **K-Nearest Neighbours** | 0.6222 ± 0.0207 | 0.7059 ± 0.0155 | 0.6914 ± 0.0179 | 6.29 s | `metric='manhattan', n_neighbors=5, weights='uniform'` |
| **Gaussian Naive Bayes** | 0.2383 ± 0.0160 | 0.2563 ± 0.0156 | 0.1895 ± 0.0160 | 0.49 s | `var_smoothing=1e-06` |

---

## 🗂️ Project Architecture & Directory Structure

```text
d:\Grade_Gaurd\
├── .gitignore                          # Clean git exclusions
├── .streamlit/
│   └── config.toml                     # Restrained navy/teal UI theme configuration
├── requirements.txt                    # Minimal exact production dependencies
├── README.md                           # Comprehensive documentation
├── app.py                              # Streamlit main entry point & navigation
├── pages/                              # Multi-page application modules
│   ├── 1_Predict_Student_Outcome.py    # Form, profile loader, CSV upload, guidance
│   ├── 2_Model_Comparison.py           # Real test metrics, CV stats, interactive charts
│   ├── 3_Dataset_Explorer.py           # Feature taxonomy, missingness, EDA charts
│   ├── 4_Methodology.py                # Pipeline diagrams, literature review table
│   └── 5_About_and_Limitations.py      # Ethics, limitations, reproduction steps
├── src/                                # Modular source library
│   ├── __init__.py
│   ├── ingest.py                       # Automated UCI dataset downloader
│   ├── data_processing.py              # Stratified split, validation, ColumnTransformer
│   ├── train.py                        # Stratified 5-Fold CV hyperparameter tuning
│   ├── evaluate.py                     # Held-out test evaluation & metric export
│   ├── predict.py                      # Single/batch inference & supportive guidance engine
│   ├── visualization.py                # Publication figure generation
│   ├── generate_report.py              # python-docx 10-page report generator
│   └── generate_pdf_report.py          # ReportLab verified 10-page PDF generator
├── data/
│   ├── raw/
│   │   └── dataset.csv                 # Official UCI ID 697 dataset copy (4,424 rows)
│   ├── processed/
│   │   ├── train_data.csv              # Stratified 80% training set (3,539 rows)
│   │   ├── test_data.csv               # Stratified 20% test set (885 rows)
│   │   ├── feature_metadata.json       # Feature taxonomy, bounds, human-readable labels
│   │   └── class_mapping.json          # Target encoding dictionary
│   └── sample_prediction_template.csv  # 1-row CSV template for test uploads
├── models/                             # Saved joblib pipeline artifacts
│   ├── best_model.joblib               # Top-performing pipeline (Random Forest)
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── support_vector_machine.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbours.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── model_metadata.json             # Tuning metadata & parameter summaries
├── reports/                            # Evaluation reports, figures, and viva prep
│   ├── figures/                        # Publication-quality charts (300 DPI)
│   │   ├── class_distribution.png
│   │   ├── model_metric_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance.png
│   │   └── pr_curves.png
│   ├── metrics_summary.csv             # Held-out test metrics table
│   ├── detailed_per_class_metrics.csv  # Per-class breakdown
│   ├── cv_results.csv                  # Cross-validation table
│   ├── evaluation_results.json         # Complete evaluation dictionary
│   ├── GradeGuard_Report.docx          # Complete 10-page research report (.docx)
│   ├── GradeGuard_Report.pdf           # Programmatically verified 10-page PDF
│   └── Viva_Questions.md               # 25+ viva questions, pitch & prediction walkthrough
└── tests/                              # Automated pytest suite (14 assertions)
    ├── __init__.py
    ├── test_data_processing.py
    ├── test_train.py
    ├── test_evaluate.py
    └── test_predict.py
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone & Set Up Environment
```bash
# Clone repository
git clone https://github.com/<your-username>/GradeGuard.git
cd GradeGuard

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Full Pipeline Reproduction (Deterministic)
```bash
# Step 1: Download official UCI dataset ID 697
python src/ingest.py

# Step 2: Clean, validate, and execute stratified 80/20 split
python src/data_processing.py

# Step 3: Run Stratified 5-Fold Cross-Validation hyperparameter tuning (all 6 models)
python src/train.py

# Step 4: Evaluate fitted pipelines on the untouched held-out test set
python src/evaluate.py

# Step 5: Generate all 6 publication figures in reports/figures/
python src/visualization.py

# Step 6: Generate the 10-page research reports (DOCX and PDF)
python src/generate_report.py
python src/generate_pdf_report.py
```

### 3. Run Automated Tests
```bash
pytest -v
```
*All 14 tests will execute and pass in approximately 2.6 seconds.*

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deployment Guidelines

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Select your repository, set **Main file path** to `app.py`, and click **Deploy**.
4. The application will build automatically using `requirements.txt` and `.streamlit/config.toml`.

---

## 🌿 Sensible Git Commit Plan

```bash
# Initialize git repository
git init

# Step 1: Baseline repository configuration
git add .gitignore requirements.txt README.md .streamlit/
git commit -m "chore: initialize repository structure, config, and requirements"

# Step 2: Data ingestion and leak-free preprocessing
git add src/ingest.py src/data_processing.py data/
git commit -m "feat(data): implement UCI dataset ingestion and leak-free stratified preprocessing"

# Step 3: Model training and evaluation suite
git add src/train.py src/evaluate.py src/predict.py models/
git commit -m "feat(models): implement Stratified 5-Fold CV hyperparameter tuning and evaluation"

# Step 4: Visualization and report generation
git add src/visualization.py src/generate_report.py src/generate_pdf_report.py reports/
git commit -m "feat(reports): generate publication figures, viva prep notes, and 10-page reports"

# Step 5: Streamlit user interface
git add app.py pages/
git commit -m "feat(ui): develop multi-page Streamlit application with restrained navy/teal theme"

# Step 6: Automated testing suite
git add tests/
git commit -m "test: add automated unit tests for preprocessing, models, and predictions"
```

---

## ⚖️ Ethical Principles & Prohibitions

1. **No Automated Punitive Administrative Actions:** GradeGuard predictions must **never** be used to withdraw scholarships, deny course registration, or penalize students.
2. **Mandatory Human Advisor Review:** Predictions are designed solely as a proactive indicator for human counselors to conduct compassionate, confidential academic check-ins.
3. **Non-Causality:** Statistical correlation in historical registry data does not imply direct causation. Root causes behind student challenges (health, financial shock, family crisis) require holistic human inquiry.

---

## 📜 Official Dataset Citation

```bibtex
@article{realinho2021predicting,
  title={Predicting Student Dropout and Academic Success},
  author={Realinho, Valentim and Machado, Jorge and Baptista, Lu{\'\i}s and Martins, M{\'o}nica V},
  journal={Data},
  volume={6},
  number={11},
  pages={110},
  year={2021},
  publisher={MDPI},
  doi={10.3390/data6110110}
}
```
*UCI Machine Learning Repository Dataset ID 697, DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).*

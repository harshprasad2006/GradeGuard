"""
GradeGuard - Page 5: About & Limitations
System boundaries, algorithmic constraints, ethical safeguards, and deployment documentation.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath("."))

st.set_page_config(page_title="About & Limitations - GradeGuard", page_icon="⚖️", layout="wide")

st.title("⚖️ About, Limitations & Ethical Guardrails")
st.markdown("""
A transparent, rigorous discussion of system constraints, data scope, institutional boundaries, 
and ethical responsibilities governing the use of machine learning in student retention.
""")

st.divider()

# Core System Limitations
st.markdown("### 1. Known Technical & Methodological Limitations")

col_lim1, col_lim2 = st.columns(2)

with col_lim1:
    st.markdown("#### ⏳ End-of-First-Year Feature Horizon")
    st.markdown("""
    The dataset includes curricular units credited, enrolled, evaluated, and approved from both the **1st and 2nd semesters**. 
    Therefore:
    - This prototype evaluates student standing **after one full year of academic coursework**.
    - It is **not** an enrollment-day or admissions-time early-warning classifier.
    - True day-one early warning models must exclude all post-enrollment curricular variables and rely strictly on high school admission metrics, which typically yields lower predictive accuracy.
    """)

    st.markdown("#### 🔄 Ambiguity of the 'Enrolled' Status")
    st.markdown("""
    Unlike `Graduate` and `Dropout`, which are terminal academic states:
    - `Enrolled` represents an active, ongoing student status.
    - Students in this category frequently resemble high-performing future graduates who took fewer credits, or at-risk dropouts who remain officially registered.
    - Consequently, all machine learning models exhibited their lowest F1 score on this class (~0.49–0.51), reflecting true sociological ambiguity.
    """)

with col_lim2:
    st.markdown("#### 🏛️ Institutional & Geographic Specificity")
    st.markdown("""
    The data originates from the **Instituto Politécnico de Portalegre**, a Portuguese polytechnic higher education institution:
    - Grading scales (0 to 20), curriculum structure (ECTS credits), and tuition policies reflect the Portuguese educational framework.
    - Direct transferability to liberal arts colleges, large research universities, or non-European higher education systems cannot be assumed without local retraining and recalibration.
    """)

    st.markdown("#### 📐 Uncalibrated Softmax Probabilities")
    st.markdown("""
    The probabilities displayed in the interface are raw softmax scores from Logistic Regression / SVM or leaf frequency proportions from Random Forest.
    - They have **not** undergone empirical probability calibration (e.g., Platt scaling or isotonic regression).
    - They must be treated as **ordinal relative confidence measures**, not actuarial probabilities of dropout risk.
    """)

st.divider()

# Ethical Principles and Responsible Use
st.markdown("### 2. Ethical Principles & Anti-Harm Guardrails")

st.markdown("""
Machine learning tools in educational institutions must operate under strict ethical governance to protect student welfare:
""")

with st.expander("🛡️ Absolute Prohibition on Automated Punitive Decisions", expanded=True):
    st.error("""
    **Mandatory Operational Rule:**
    GradeGuard outputs must **never** be used to automate punitive, restrictive, or gatekeeping decisions. 
    Examples of prohibited uses include:
    - Automatically revoking financial aid or scholarships based on a predicted dropout risk.
    - Restricting a student from enrolling in upper-division coursework.
    - Flagging students on public or faculty rosters with stigmatizing risk labels.
    """)

with st.expander("🤝 Human-in-the-Loop Academic Advising", expanded=True):
    st.success("""
    **Supportive Decision-Support Mandate:**
    Predictions are designed strictly to alert professional academic advisors to consider proactive, non-punitive outreach:
    - Offering voluntary tutoring, study skills coaching, or degree roadmap planning.
    - Connecting students with emergency financial aid counseling if administrative flags exist.
    - Conducting empathetic, confidential 1-on-1 check-ins where human factors (family, health, motivation) take precedence.
    """)

with st.expander("⚖️ Demographic Features & Fairness", expanded=False):
    st.warning("""
    **Fairness Considerations:**
    The UCI dataset contains demographic attributes (Gender, Age at Enrollment, Marital Status, Parental Qualifications). 
    While historical data reflects societal inequalities (such as gendered enrollment patterns or generational educational gaps), 
    advising interventions must focus strictly on actionable academic and institutional support rather than demographic stereotypes.
    """)

st.divider()

# Technical Environment & Reproduction
st.markdown("### 3. Software Environment & Execution Commands")

st.markdown("""
| Component | Specification |
| :--- | :--- |
| **Language** | Python 3.13 |
| **Frameworks** | scikit-learn, pandas, numpy, matplotlib, streamlit |
| **Dataset** | UCI Machine Learning Repository (ID 697, DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)) |
| **Primary Metric** | Macro F1 (Stratified 5-Fold Cross-Validation) |
| **Tested Platforms** | Windows 11 / Linux / Streamlit Community Cloud |
""")

st.code("""
# 1. Clone repository and install dependencies
pip install -r requirements.txt

# 2. Ingest official UCI dataset (saves to data/raw/dataset.csv)
python src/ingest.py

# 3. Clean and split data (saves to data/processed/)
python src/data_processing.py

# 4. Train 6 classical ML models with 5-fold Stratified CV
python src/train.py

# 5. Evaluate on held-out test cohort
python src/evaluate.py

# 6. Generate all figures
python src/visualization.py

# 7. Run automated test suite
pytest -v

# 8. Launch Streamlit application
streamlit run app.py
""", language="bash")

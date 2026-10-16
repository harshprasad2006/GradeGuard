"""
GradeGuard: Student Academic Success and Dropout Risk Prediction
Main Application Entry Point.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath("."))
from src.data_processing import TARGET_COLUMN

st.set_page_config(
    page_title="GradeGuard - Student Academic Success & Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header Section
st.title("🎓 GradeGuard")
st.subheader("Student Academic Success and Dropout Risk Prediction Prototype")

st.markdown("""
Welcome to **GradeGuard**, a machine learning decision-support system designed to analyze student academic trajectory indicators 
and predict one of three academic outcomes: **Dropout**, **Enrolled**, or **Graduate**. 
Built on the peer-reviewed **UCI Machine Learning Repository Dataset ID 697** (*Predict Students' Dropout and Academic Success*, DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)), 
this project compares six classical machine learning algorithms through leak-free pipelines and stratified cross-validation.
""")

st.divider()

# High-Level Summary Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Student Records", value="4,424", help="Official UCI dataset records (2008/09 to 2018/19)")

with col2:
    st.metric(label="Features Analyzed", value="36", help="Demographic, academic, socio-economic & macroeconomic attributes")

with col3:
    st.metric(label="Target Outcomes", value="3 Classes", help="Dropout (32.1%), Enrolled (18.0%), Graduate (49.9%)")

with col4:
    # Load best model test macro F1 if available
    metrics_path = os.path.join("reports", "metrics_summary.csv")
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        best_f1 = m_df.iloc[0]["Macro_F1"]
        best_name = m_df.iloc[0]["Model"]
        st.metric(label=f"Top Model ({best_name})", value=f"{best_f1:.4f} F1", help="Held-out test set Macro F1 score")
    else:
        st.metric(label="Top Model", value="Random Forest", help="Trained with Stratified 5-Fold CV")

st.divider()

# Important Methodological Disclaimers
st.markdown("### ⚠️ Critical Methodological Context & Scope")

with st.expander("📌 End-of-Normal-Duration vs. Early-Warning Dataset", expanded=True):
    st.info("""
    **Temporal Scope Notice:** 
    The UCI dataset contains curricular unit counts and examination averages from both the **1st and 2nd semesters**. 
    Consequently, this model evaluates student status at the **conclusion of the first academic year** (or normal degree duration). 
    It does **not** automatically validate a day-one admissions early-warning prediction. Genuine enrollment-time screening would require 
    restricting inputs strictly to pre-entry demographic and high-school admission attributes.
    """)

with st.expander("⚖️ Decision-Support Prototype vs. Autonomous Decision Making", expanded=False):
    st.warning("""
    **Ethical & Administrative Guardrails:**
    GradeGuard is an **educational decision-support prototype**, not a definitive judgment of any student's potential, intellect, or future. 
    Machine learning correlations in historical data do **not** represent causal mechanisms. 
    Under no circumstances should automated model outputs be used to withdraw aid, deny course enrollment, or penalize students. 
    Qualified human academic advisors and counselors must always review context holistically before initiating supportive interventions.
    """)

with st.expander("📊 Probability Calibration Notice", expanded=False):
    st.markdown("""
    **Softmax Probability Disclaimer:**
    Output scores displayed in the prediction interface represent model-derived softmax or tree class frequencies. 
    Because empirical calibration (e.g., Platt scaling or isotonic regression) was not independently fitted and validated across subgroups, 
    these percentages should be interpreted as **relative confidence scores**, not exact real-world risk probabilities.
    """)

st.divider()

# Feature Highlights & Navigation Guide
st.markdown("### 🧭 System Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("#### 🔮 [Predict Student Outcome](Predict_Student_Outcome)")
    st.markdown("""
    - Interactive 36-feature input form with human-readable labels.
    - Preloaded sample student profiles (Graduate, Dropout, Enrolled).
    - Advanced CSV batch upload using exact training schema.
    - Model selection & supportive advising recommendations.
    """)

with nav_col2:
    st.markdown("#### 📈 [Model Comparison](Model_Comparison)")
    st.markdown("""
    - Dynamic comparison table across 6 classical ML classifiers.
    - Real held-out test metrics & 5-fold CV statistics.
    - Multi-class Confusion Matrices, ROC-AUC, and PR curves.
    - Feature importance analysis and CSV downloads.
    """)

with nav_col3:
    st.markdown("#### 🔍 [Dataset Explorer](Dataset_Explorer)")
    st.markdown("""
    - Demographic and curricular unit distribution charts.
    - Missingness, duplicates, and class imbalance inspection.
    - Detailed feature summaries and raw data table preview.
    - Provenance and collection methodology documentation.
    """)

st.markdown("""
---
**GradeGuard ML Mini-Project** | Developed with scikit-learn, pandas, and Streamlit.  
Dataset Citation: Realinho et al. (2021), *Data*, 6(11), 110. DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
""")

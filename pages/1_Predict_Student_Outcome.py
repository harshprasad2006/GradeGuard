"""
GradeGuard - Page 1: Predict Student Outcome
Interactive prediction interface with robust session state management,
instant test-case buttons, direct CSV upload prediction, and supportive guidance.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.abspath("."))
from src.predict import (
    predict_student,
    load_metadata,
    validate_student_input,
    export_sample_csv_template,
)
from src.data_processing import (
    TARGET_COLUMN,
    HUMAN_READABLE_LABELS,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
)

st.set_page_config(page_title="Predict Student Outcome - GradeGuard", page_icon="🔮", layout="wide")

st.title("🔮 Predict Student Academic Outcome")
st.markdown("""
Evaluate higher education student trajectory indicators to predict one of three academic outcomes: 
**Dropout**, **Enrolled**, or **Graduate**.  
*All predictions are generated directly from the trained classical ML pipelines using official UCI Dataset 697 attributes.*
""")

# Load metadata
metadata = load_metadata()
all_features = metadata["all_features_in_order"]

# Load authentic test records from held-out test data
test_df = pd.read_csv(os.path.join("data", "processed", "test_data.csv"))

# Benchmark profile definitions
benchmark_cases = {
    "Case 1: High Performer (Graduate)": {
        "record": test_df[
            (test_df[TARGET_COLUMN] == "Graduate") & 
            (test_df["Curricular units 1st sem (approved)"] >= 5) & 
            (test_df["Curricular units 2nd sem (approved)"] >= 5)
        ].iloc[0].drop(TARGET_COLUMN).to_dict(),
        "expected": "Graduate (~79% confidence)",
        "desc": "Passed all 6/6 units in both semesters, high grades, tuition current.",
    },
    "Case 2: Severe Dropout Risk (Dropout)": {
        "record": test_df[
            (test_df[TARGET_COLUMN] == "Dropout") & 
            (test_df["Curricular units 1st sem (approved)"] == 0) & 
            (test_df["Curricular units 2nd sem (approved)"] == 0)
        ].iloc[0].drop(TARGET_COLUMN).to_dict(),
        "expected": "Dropout (~70% confidence)",
        "desc": "Passed 0/5 units in both semesters, acute credit failure.",
    },
    "Case 3: Active Ongoing Student (Enrolled)": {
        "record": test_df[
            (test_df[TARGET_COLUMN] == "Enrolled") & 
            (test_df["Curricular units 2nd sem (approved)"] >= 3)
        ].iloc[0].drop(TARGET_COLUMN).to_dict(),
        "expected": "Enrolled (~60% confidence)",
        "desc": "Passed 3-5 units, still actively enrolled and progressing.",
    },
    "Case 4: Sudden 2nd Sem Slump (Dropout)": {
        "record": test_df[
            (test_df[TARGET_COLUMN] == "Dropout") & 
            (test_df["Curricular units 1st sem (approved)"] >= 5) & 
            (test_df["Curricular units 2nd sem (approved)"] <= 1)
        ].iloc[0].drop(TARGET_COLUMN).to_dict(),
        "expected": "Dropout (~67% confidence)",
        "desc": "Passed 6 units in Sem 1, but failed all except 1 unit in Sem 2.",
    },
}

# Initialize session state for student record
if "active_student" not in st.session_state:
    st.session_state["active_student"] = benchmark_cases["Case 1: High Performer (Graduate)"]["record"].copy()
if "last_case_loaded" not in st.session_state:
    st.session_state["last_case_loaded"] = "Case 1: High Performer (Graduate)"

# Sidebar Model Configuration
st.sidebar.header("⚙️ Model Configuration")
available_models = {
    "Best Model (Random Forest)": "models/best_model.joblib",
    "Logistic Regression": "models/logistic_regression.joblib",
    "Support Vector Machine (SVM)": "models/support_vector_machine.joblib",
    "Decision Tree Classifier": "models/decision_tree.joblib",
    "K-Nearest Neighbours": "models/k_nearest_neighbours.joblib",
    "Gaussian Naive Bayes": "models/gaussian_naive_bayes.joblib",
}

selected_model_name = st.sidebar.selectbox("Active Inference Model", list(available_models.keys()), index=0)
selected_model_path = available_models[selected_model_name]

st.sidebar.divider()
st.sidebar.markdown("### 📥 Download Prediction Template")
template_path = "data/sample_prediction_template.csv"
if not os.path.exists(template_path):
    export_sample_csv_template(template_path)

with open(template_path, "rb") as f:
    st.sidebar.download_button(
        label="Download 1-Row CSV Template",
        data=f,
        file_name="gradeguard_student_template.csv",
        mime="text/csv",
        help="Download official column headers and valid sample values for batch testing",
    )

# -------------------------------------------------------------
# QUICK TEST CASE SELECTOR (Direct One-Click Verification)
# -------------------------------------------------------------
st.markdown("### ⚡ Quick Model Verification (Select a Benchmark Profile)")
st.caption("Click any profile below to instantly load different student trajectories and verify distinct model outputs:")

quick_cols = st.columns(4)

with quick_cols[0]:
    if st.button("🟢 Case 1: High Performer\n(Expected: Graduate)", use_container_width=True):
        st.session_state["active_student"] = benchmark_cases["Case 1: High Performer (Graduate)"]["record"].copy()
        st.session_state["last_case_loaded"] = "Case 1: High Performer (Graduate)"
        st.rerun()

with quick_cols[1]:
    if st.button("🔴 Case 2: Dropout Risk\n(Expected: Dropout)", use_container_width=True):
        st.session_state["active_student"] = benchmark_cases["Case 2: Severe Dropout Risk (Dropout)"]["record"].copy()
        st.session_state["last_case_loaded"] = "Case 2: Severe Dropout Risk (Dropout)"
        st.rerun()

with quick_cols[2]:
    if st.button("🟡 Case 3: Active Enrolled\n(Expected: Enrolled)", use_container_width=True):
        st.session_state["active_student"] = benchmark_cases["Case 3: Active Ongoing Student (Enrolled)"]["record"].copy()
        st.session_state["last_case_loaded"] = "Case 3: Active Ongoing Student (Enrolled)"
        st.rerun()

with quick_cols[3]:
    if st.button("🟠 Case 4: Sem 2 Slump\n(Expected: Dropout)", use_container_width=True):
        st.session_state["active_student"] = benchmark_cases["Case 4: Sudden 2nd Sem Slump (Dropout)"]["record"].copy()
        st.session_state["last_case_loaded"] = "Case 4: Sudden 2nd Sem Slump (Dropout)"
        st.rerun()

st.info(f"📌 **Currently Active Profile:** `{st.session_state['last_case_loaded']}`")

st.divider()

# -------------------------------------------------------------
# INPUT MODE: Interactive Form vs Upload CSV
# -------------------------------------------------------------
input_mode = st.radio(
    "Choose how you want to adjust or supply student attributes:",
    ["Interactive Attribute Editor", "Upload 1-Row CSV File"],
    horizontal=True,
)

if input_mode == "Upload 1-Row CSV File":
    st.markdown("#### Upload Student Record CSV")
    uploaded_file = st.file_uploader("Upload a 1-row CSV file containing all 36 feature columns", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if uploaded_df.empty:
                st.error("Uploaded CSV file is empty.")
            else:
                is_valid, issues = validate_student_input(uploaded_df)
                if not is_valid:
                    st.error(f"CSV Validation Errors: {'; '.join(issues)}")
                else:
                    st.session_state["active_student"] = uploaded_df.iloc[0].to_dict()
                    st.session_state["last_case_loaded"] = f"Uploaded File: {uploaded_file.name}"
                    st.success("Successfully loaded and validated student record from CSV!")
                    st.dataframe(uploaded_df.head(1), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

else:
    # Interactive Editor Mode with Tabs
    curr = st.session_state["active_student"]

    tab_academic, tab_admission, tab_demo, tab_socio = st.tabs([
        "📚 Curricular Performance (Sem 1 & 2)",
        "🎯 Admission & Course",
        "👤 Demographics & Personal",
        "💶 Socio-Economic & Macro",
    ])

    with tab_academic:
        st.markdown("#### Curricular Units & Grades (End-of-Year Academic Indicators)")
        st.caption("Change units approved or grades below to see how the model responds in real time:")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1st Semester Indicators")
            curr["Curricular units 1st sem (enrolled)"] = st.number_input(
                "1st Sem Units Enrolled", min_value=0, max_value=30,
                value=int(curr.get("Curricular units 1st sem (enrolled)", 6)),
                key="input_sem1_enr",
            )
            curr["Curricular units 1st sem (approved)"] = st.number_input(
                "1st Sem Units Approved", min_value=0, max_value=30,
                value=int(curr.get("Curricular units 1st sem (approved)", 6)),
                key="input_sem1_appr",
            )
            curr["Curricular units 1st sem (evaluations)"] = st.number_input(
                "1st Sem Units Evaluated", min_value=0, max_value=40,
                value=int(curr.get("Curricular units 1st sem (evaluations)", 6)),
                key="input_sem1_eval",
            )
            curr["Curricular units 1st sem (grade)"] = st.number_input(
                "1st Sem Grade Average (0 - 20 Scale)", min_value=0.0, max_value=20.0,
                value=float(curr.get("Curricular units 1st sem (grade)", 13.5)), step=0.1,
                key="input_sem1_grd",
            )
            curr["Curricular units 1st sem (credited)"] = st.number_input(
                "1st Sem Units Credited", min_value=0, max_value=25,
                value=int(curr.get("Curricular units 1st sem (credited)", 0)),
                key="input_sem1_cred",
            )
            curr["Curricular units 1st sem (without evaluations)"] = st.number_input(
                "1st Sem Units Without Evaluations", min_value=0, max_value=15,
                value=int(curr.get("Curricular units 1st sem (without evaluations)", 0)),
                key="input_sem1_noeval",
            )

        with col2:
            st.markdown("##### 2nd Semester Indicators")
            curr["Curricular units 2nd sem (enrolled)"] = st.number_input(
                "2nd Sem Units Enrolled", min_value=0, max_value=30,
                value=int(curr.get("Curricular units 2nd sem (enrolled)", 6)),
                key="input_sem2_enr",
            )
            curr["Curricular units 2nd sem (approved)"] = st.number_input(
                "2nd Sem Units Approved", min_value=0, max_value=30,
                value=int(curr.get("Curricular units 2nd sem (approved)", 6)),
                key="input_sem2_appr",
            )
            curr["Curricular units 2nd sem (evaluations)"] = st.number_input(
                "2nd Sem Units Evaluated", min_value=0, max_value=40,
                value=int(curr.get("Curricular units 2nd sem (evaluations)", 6)),
                key="input_sem2_eval",
            )
            curr["Curricular units 2nd sem (grade)"] = st.number_input(
                "2nd Sem Grade Average (0 - 20 Scale)", min_value=0.0, max_value=20.0,
                value=float(curr.get("Curricular units 2nd sem (grade)", 13.8)), step=0.1,
                key="input_sem2_grd",
            )
            curr["Curricular units 2nd sem (credited)"] = st.number_input(
                "2nd Sem Units Credited", min_value=0, max_value=25,
                value=int(curr.get("Curricular units 2nd sem (credited)", 0)),
                key="input_sem2_cred",
            )
            curr["Curricular units 2nd sem (without evaluations)"] = st.number_input(
                "2nd Sem Units Without Evaluations", min_value=0, max_value=15,
                value=int(curr.get("Curricular units 2nd sem (without evaluations)", 0)),
                key="input_sem2_noeval",
            )

    with tab_admission:
        col1, col2 = st.columns(2)
        with col1:
            course_labels = HUMAN_READABLE_LABELS["Course"]
            curr_course = int(curr.get("Course", 9500))
            course_keys = list(course_labels.keys())
            course_idx = course_keys.index(curr_course) if curr_course in course_keys else 0
            curr["Course"] = st.selectbox(
                "Undergraduate Degree Program",
                course_keys,
                format_func=lambda k: f"{course_labels.get(k, k)} (Code {k})",
                index=course_idx,
                key="input_course",
            )

            app_mode_labels = HUMAN_READABLE_LABELS["Application mode"]
            curr_app = int(curr.get("Application mode", 1))
            app_keys = list(app_mode_labels.keys())
            app_idx = app_keys.index(curr_app) if curr_app in app_keys else 0
            curr["Application mode"] = st.selectbox(
                "Application Mode",
                app_keys,
                format_func=lambda k: f"{app_mode_labels.get(k, k)}",
                index=app_idx,
                key="input_app_mode",
            )

            curr["Application order"] = st.slider(
                "Application Choice Order (0 = 1st choice, 9 = 10th choice)",
                min_value=0, max_value=9,
                value=int(curr.get("Application order", 1)),
                key="input_app_order",
            )

        with col2:
            prev_qual_labels = HUMAN_READABLE_LABELS["Previous qualification"]
            curr_qual = int(curr.get("Previous qualification", 1))
            qual_keys = list(prev_qual_labels.keys())
            qual_idx = qual_keys.index(curr_qual) if curr_qual in qual_keys else 0
            curr["Previous qualification"] = st.selectbox(
                "Previous Qualification Level",
                qual_keys,
                format_func=lambda k: f"{prev_qual_labels.get(k, k)}",
                index=qual_idx,
                key="input_prev_qual",
            )

            curr["Previous qualification (grade)"] = st.number_input(
                "Previous Qualification Grade (0 - 200 Scale)", min_value=0.0, max_value=200.0,
                value=float(curr.get("Previous qualification (grade)", 130.0)), step=0.5,
                key="input_prev_grd",
            )
            curr["Admission grade"] = st.number_input(
                "Institutional Admission Grade (0 - 200 Scale)", min_value=0.0, max_value=200.0,
                value=float(curr.get("Admission grade", 128.5)), step=0.5,
                key="input_adm_grd",
            )

    with tab_demo:
        col1, col2 = st.columns(2)
        with col1:
            curr["Age at enrollment"] = st.number_input(
                "Age at Enrollment (Years)", min_value=17, max_value=75,
                value=int(curr.get("Age at enrollment", 20)),
                key="input_age",
            )
            gender_val = int(curr.get("Gender", 1))
            curr["Gender"] = st.selectbox(
                "Gender", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female",
                index=0 if gender_val == 1 else 1,
                key="input_gender",
            )
            marital_labels = HUMAN_READABLE_LABELS["Marital Status"]
            curr_marital = int(curr.get("Marital Status", 1))
            m_keys = list(marital_labels.keys())
            m_idx = m_keys.index(curr_marital) if curr_marital in m_keys else 0
            curr["Marital Status"] = st.selectbox(
                "Marital Status", m_keys,
                format_func=lambda k: marital_labels.get(k, str(k)),
                index=m_idx,
                key="input_marital",
            )
            attend_val = int(curr.get("Daytime/evening attendance", 1))
            curr["Daytime/evening attendance"] = st.selectbox(
                "Attendance Schedule", [1, 0],
                format_func=lambda v: "Daytime" if v == 1 else "Evening",
                index=0 if attend_val == 1 else 1,
                key="input_attend",
            )

        with col2:
            disp_val = int(curr.get("Displaced", 0))
            curr["Displaced"] = st.selectbox(
                "Displaced Student (Relocated from home)", [1, 0],
                format_func=lambda v: "Yes" if v == 1 else "No",
                index=0 if disp_val == 1 else 1,
                key="input_displaced",
            )
            needs_val = int(curr.get("Educational special needs", 0))
            curr["Educational special needs"] = st.selectbox(
                "Educational Special Needs", [1, 0],
                format_func=lambda v: "Yes" if v == 1 else "No",
                index=0 if needs_val == 1 else 1,
                key="input_needs",
            )
            intl_val = int(curr.get("International", 0))
            curr["International"] = st.selectbox(
                "International Student", [1, 0],
                format_func=lambda v: "Yes" if v == 1 else "No",
                index=0 if intl_val == 1 else 1,
                key="input_intl",
            )
            nac_labels = HUMAN_READABLE_LABELS["Nacionality"]
            curr_nac = int(curr.get("Nacionality", 1))
            n_keys = list(nac_labels.keys())
            n_idx = n_keys.index(curr_nac) if curr_nac in n_keys else 0
            curr["Nacionality"] = st.selectbox(
                "Nationality", n_keys,
                format_func=lambda k: nac_labels.get(k, str(k)),
                index=n_idx,
                key="input_nac",
            )

    with tab_socio:
        col1, col2 = st.columns(2)
        with col1:
            tuition_val = int(curr.get("Tuition fees up to date", 1))
            curr["Tuition fees up to date"] = st.selectbox(
                "Tuition Fees Up to Date", [1, 0],
                format_func=lambda v: "Yes (Current)" if v == 1 else "No (Overdue)",
                index=0 if tuition_val == 1 else 1,
                key="input_tuition",
            )
            debt_val = int(curr.get("Debtor", 0))
            curr["Debtor"] = st.selectbox(
                "Administrative Debtor Status", [1, 0],
                format_func=lambda v: "Yes (Has Unpaid Debt)" if v == 1 else "No (Clear)",
                index=0 if debt_val == 1 else 1,
                key="input_debtor",
            )
            schol_val = int(curr.get("Scholarship holder", 0))
            curr["Scholarship holder"] = st.selectbox(
                "Scholarship Recipient", [1, 0],
                format_func=lambda v: "Yes (Scholarship Awarded)" if v == 1 else "No",
                index=0 if schol_val == 1 else 1,
                key="input_schol",
            )
            curr["Mother's qualification"] = int(curr.get("Mother's qualification", 1))
            curr["Father's qualification"] = int(curr.get("Father's qualification", 1))
            curr["Mother's occupation"] = int(curr.get("Mother's occupation", 5))
            curr["Father's occupation"] = int(curr.get("Father's occupation", 5))

        with col2:
            st.markdown("##### Regional Macroeconomic Context")
            curr["Unemployment rate"] = st.number_input(
                "Unemployment Rate (%)", min_value=0.0, max_value=30.0,
                value=float(curr.get("Unemployment rate", 11.1)), step=0.1,
                key="input_unemp",
            )
            curr["Inflation rate"] = st.number_input(
                "Inflation Rate (%)", min_value=-5.0, max_value=20.0,
                value=float(curr.get("Inflation rate", 1.4)), step=0.1,
                key="input_infl",
            )
            curr["GDP"] = st.number_input(
                "GDP Growth Rate (%)", min_value=-15.0, max_value=15.0,
                value=float(curr.get("GDP", 1.74)), step=0.01,
                key="input_gdp",
            )

    st.session_state["active_student"] = curr

st.divider()

# -------------------------------------------------------------
# PREDICTION EXECUTION
# -------------------------------------------------------------
st.markdown("### 🚀 Model Inference")

active_data = st.session_state["active_student"]

# Display key summary indicators
sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
with sum_c1:
    st.metric("Sem 1 Units Approved", f"{int(active_data.get('Curricular units 1st sem (approved)', 0))} / {int(active_data.get('Curricular units 1st sem (enrolled)', 0))}")
with sum_c2:
    st.metric("Sem 2 Units Approved", f"{int(active_data.get('Curricular units 2nd sem (approved)', 0))} / {int(active_data.get('Curricular units 2nd sem (enrolled)', 0))}")
with sum_c3:
    t_status = "Paid (Current)" if active_data.get("Tuition fees up to date", 1) == 1 else "OVERDUE"
    st.metric("Tuition Status", t_status)
with sum_c4:
    st.metric("Student Age", f"{int(active_data.get('Age at enrollment', 20))} yrs")

if st.button("🚀 Analyze Student Academic Trajectory", type="primary", use_container_width=True):
    try:
        with st.spinner(f"Computing prediction with {selected_model_name}..."):
            result = predict_student(active_data, model_path=selected_model_path)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if predicted_class == "Graduate":
                st.success(f"### Predicted Outcome: **{predicted_class}**")
                st.markdown("The student displays positive academic progression consistent with degree completion.")
            elif predicted_class == "Dropout":
                st.error(f"### Predicted Outcome: **{predicted_class}** Risk")
                st.markdown("Credit deficits, failed evaluations, or administrative flags indicate elevated dropout risk.")
            else:
                st.info(f"### Predicted Outcome: **{predicted_class}** (Active / In-Progress)")
                st.markdown("The student remains actively enrolled with an intermediate progression trajectory.")

            st.metric(f"Model Confidence ({selected_model_name})", f"{confidence:.1%}")
            st.caption("Note: Represents model-derived softmax class score, not a calibrated real-world probability.")

        with res_col2:
            st.markdown("#### Outcome Probability Distribution")
            prob_df = pd.DataFrame({
                "Outcome": list(probabilities.keys()),
                "Probability": list(probabilities.values()),
            })
            st.bar_chart(prob_df.set_index("Outcome"), color="#008080", use_container_width=True)

        st.divider()

        # Supportive Advising Guidance
        st.markdown("### 🧭 Supportive Decision-Support Guidance")
        st.markdown("""
        *These non-deterministic recommendations are designed for academic advisors to guide constructive student support, 
        not automated or punitive administrative actions.*
        """)

        for rec in result["guidance"]:
            if "Tuition" in rec or "Financial" in rec:
                st.warning(f"💳 {rec}")
            elif "Credit" in rec or "Pace" in rec:
                st.info(f"📖 {rec}")
            elif "Scholarship" in rec:
                st.warning(f"🎖️ {rec}")
            elif "Roadmap" in rec or "Excellence" in rec or "Outreach" in rec:
                st.info(f"🤝 {rec}")
            else:
                st.caption(f"ℹ️ {rec}")

        st.divider()
        st.caption(f"**Disclaimer:** {result['disclaimer']}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

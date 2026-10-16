"""
GradeGuard - Page 1: Predict Student Outcome
Interactive prediction interface supporting human-readable inputs,
sample profile preloading, CSV uploads, probability visualizations, and supportive advising guidance.
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
    get_sample_profiles,
    load_metadata,
    validate_student_input,
    export_sample_csv_template,
)
from src.data_processing import (
    HUMAN_READABLE_LABELS,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
)

st.set_page_config(page_title="Predict Student Outcome - GradeGuard", page_icon="🔮", layout="wide")

st.title("🔮 Predict Student Academic Outcome")
st.markdown("""
Enter student curricular, demographic, and institutional attributes to generate a predicted academic trajectory: 
**Dropout**, **Enrolled**, or **Graduate**.  
*All inputs use official UCI repository categories and verified feature bounds.*
""")

metadata = load_metadata()
all_features = metadata["all_features_in_order"]

# Model Selection Sidebar
st.sidebar.header("⚙️ Model Configuration")
available_models = {
    "Best Model (Random Forest)": "models/best_model.joblib",
    "Random Forest Classifier": "models/random_forest.joblib",
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

# Input Mode: Form Presets vs Manual vs CSV Upload
st.markdown("### 1. Select Input Method")
input_mode = st.radio(
    "Choose how you want to provide student feature values:",
    ["Use Preset Student Profile", "Manual Interactive Form", "Upload 1-Row CSV File"],
    horizontal=True,
)

sample_profiles = get_sample_profiles()
current_data = {}

if input_mode == "Use Preset Student Profile":
    preset_choice = st.selectbox(
        "Select an authentic benchmark profile from the held-out test cohort:",
        list(sample_profiles.keys()),
    )
    current_data = sample_profiles[preset_choice].copy()
    st.info(f"Loaded feature values for **{preset_choice}**. You may review or modify parameters below before running inference.")

elif input_mode == "Upload 1-Row CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file with all 36 feature columns", type=["csv"])
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
                    current_data = uploaded_df.iloc[0].to_dict()
                    st.success("Successfully loaded and validated student record from uploaded CSV!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("Upload a CSV matching the training schema, or switch to 'Use Preset Student Profile'.")
        current_data = sample_profiles["Profile A: High Academic Momentum (Graduate Benchmark)"].copy()

else:  # Manual Interactive Form
    current_data = sample_profiles["Profile A: High Academic Momentum (Graduate Benchmark)"].copy()

st.divider()

# Interactive Form Tabs
st.markdown("### 2. Student Attributes")

tab_academic, tab_admission, tab_demo, tab_socio = st.tabs([
    "📚 Curricular Performance (Sem 1 & 2)",
    "🎯 Admission & Course",
    "👤 Demographics & Personal",
    "💶 Socio-Economic & Macro",
])

# Working feature dictionary
user_features = {}

with tab_academic:
    st.markdown("#### Curricular Units & Grades (End-of-Year Academic Indicators)")
    st.caption("These indicators represent credit pace and evaluation marks at the close of the first academic year.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1st Semester Indicators")
        user_features["Curricular units 1st sem (enrolled)"] = st.number_input(
            "1st Sem Units Enrolled", min_value=0, max_value=30,
            value=int(current_data.get("Curricular units 1st sem (enrolled)", 6)),
        )
        user_features["Curricular units 1st sem (approved)"] = st.number_input(
            "1st Sem Units Approved", min_value=0, max_value=30,
            value=int(current_data.get("Curricular units 1st sem (approved)", 6)),
        )
        user_features["Curricular units 1st sem (evaluations)"] = st.number_input(
            "1st Sem Units Evaluated", min_value=0, max_value=40,
            value=int(current_data.get("Curricular units 1st sem (evaluations)", 6)),
        )
        user_features["Curricular units 1st sem (grade)"] = st.number_input(
            "1st Sem Grade Average (0 - 20 Scale)", min_value=0.0, max_value=20.0,
            value=float(current_data.get("Curricular units 1st sem (grade)", 13.5)), step=0.1,
        )
        user_features["Curricular units 1st sem (credited)"] = st.number_input(
            "1st Sem Units Credited", min_value=0, max_value=25,
            value=int(current_data.get("Curricular units 1st sem (credited)", 0)),
        )
        user_features["Curricular units 1st sem (without evaluations)"] = st.number_input(
            "1st Sem Units Without Evaluations", min_value=0, max_value=15,
            value=int(current_data.get("Curricular units 1st sem (without evaluations)", 0)),
        )

    with col2:
        st.markdown("##### 2nd Semester Indicators")
        user_features["Curricular units 2nd sem (enrolled)"] = st.number_input(
            "2nd Sem Units Enrolled", min_value=0, max_value=30,
            value=int(current_data.get("Curricular units 2nd sem (enrolled)", 6)),
        )
        user_features["Curricular units 2nd sem (approved)"] = st.number_input(
            "2nd Sem Units Approved", min_value=0, max_value=30,
            value=int(current_data.get("Curricular units 2nd sem (approved)", 6)),
        )
        user_features["Curricular units 2nd sem (evaluations)"] = st.number_input(
            "2nd Sem Units Evaluated", min_value=0, max_value=40,
            value=int(current_data.get("Curricular units 2nd sem (evaluations)", 6)),
        )
        user_features["Curricular units 2nd sem (grade)"] = st.number_input(
            "2nd Sem Grade Average (0 - 20 Scale)", min_value=0.0, max_value=20.0,
            value=float(current_data.get("Curricular units 2nd sem (grade)", 13.8)), step=0.1,
        )
        user_features["Curricular units 2nd sem (credited)"] = st.number_input(
            "2nd Sem Units Credited", min_value=0, max_value=25,
            value=int(current_data.get("Curricular units 2nd sem (credited)", 0)),
        )
        user_features["Curricular units 2nd sem (without evaluations)"] = st.number_input(
            "2nd Sem Units Without Evaluations", min_value=0, max_value=15,
            value=int(current_data.get("Curricular units 2nd sem (without evaluations)", 0)),
        )

with tab_admission:
    col1, col2 = st.columns(2)
    with col1:
        course_labels = HUMAN_READABLE_LABELS["Course"]
        curr_course = int(current_data.get("Course", 9500))
        course_keys = list(course_labels.keys())
        course_idx = course_keys.index(curr_course) if curr_course in course_keys else 0
        selected_course = st.selectbox(
            "Undergraduate Degree Program",
            course_keys,
            format_func=lambda k: f"{course_labels.get(k, k)} (Code {k})",
            index=course_idx,
        )
        user_features["Course"] = selected_course

        app_mode_labels = HUMAN_READABLE_LABELS["Application mode"]
        curr_app = int(current_data.get("Application mode", 1))
        app_keys = list(app_mode_labels.keys())
        app_idx = app_keys.index(curr_app) if curr_app in app_keys else 0
        selected_app_mode = st.selectbox(
            "Application Mode",
            app_keys,
            format_func=lambda k: f"{app_mode_labels.get(k, k)}",
            index=app_idx,
        )
        user_features["Application mode"] = selected_app_mode

        user_features["Application order"] = st.slider(
            "Application Choice Order (0 = 1st choice, 9 = 10th choice)",
            min_value=0, max_value=9,
            value=int(current_data.get("Application order", 1)),
        )

    with col2:
        prev_qual_labels = HUMAN_READABLE_LABELS["Previous qualification"]
        curr_qual = int(current_data.get("Previous qualification", 1))
        qual_keys = list(prev_qual_labels.keys())
        qual_idx = qual_keys.index(curr_qual) if curr_qual in qual_keys else 0
        selected_qual = st.selectbox(
            "Previous Qualification Level",
            qual_keys,
            format_func=lambda k: f"{prev_qual_labels.get(k, k)}",
            index=qual_idx,
        )
        user_features["Previous qualification"] = selected_qual

        user_features["Previous qualification (grade)"] = st.number_input(
            "Previous Qualification Grade (0 - 200 Scale)", min_value=0.0, max_value=200.0,
            value=float(current_data.get("Previous qualification (grade)", 130.0)), step=0.5,
        )
        user_features["Admission grade"] = st.number_input(
            "Institutional Admission Grade (0 - 200 Scale)", min_value=0.0, max_value=200.0,
            value=float(current_data.get("Admission grade", 128.5)), step=0.5,
        )

with tab_demo:
    col1, col2 = st.columns(2)
    with col1:
        user_features["Age at enrollment"] = st.number_input(
            "Age at Enrollment (Years)", min_value=17, max_value=75,
            value=int(current_data.get("Age at enrollment", 20)),
        )
        gender_val = int(current_data.get("Gender", 1))
        user_features["Gender"] = st.selectbox(
            "Gender", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female",
            index=0 if gender_val == 1 else 1,
        )
        marital_labels = HUMAN_READABLE_LABELS["Marital Status"]
        curr_marital = int(current_data.get("Marital Status", 1))
        m_keys = list(marital_labels.keys())
        m_idx = m_keys.index(curr_marital) if curr_marital in m_keys else 0
        user_features["Marital Status"] = st.selectbox(
            "Marital Status", m_keys,
            format_func=lambda k: marital_labels.get(k, str(k)),
            index=m_idx,
        )
        attend_val = int(current_data.get("Daytime/evening attendance", 1))
        user_features["Daytime/evening attendance"] = st.selectbox(
            "Attendance Schedule", [1, 0],
            format_func=lambda v: "Daytime" if v == 1 else "Evening",
            index=0 if attend_val == 1 else 1,
        )

    with col2:
        disp_val = int(current_data.get("Displaced", 0))
        user_features["Displaced"] = st.selectbox(
            "Displaced Student (Relocated from home)", [1, 0],
            format_func=lambda v: "Yes" if v == 1 else "No",
            index=0 if disp_val == 1 else 1,
        )
        needs_val = int(current_data.get("Educational special needs", 0))
        user_features["Educational special needs"] = st.selectbox(
            "Educational Special Needs", [1, 0],
            format_func=lambda v: "Yes" if v == 1 else "No",
            index=0 if needs_val == 1 else 1,
        )
        intl_val = int(current_data.get("International", 0))
        user_features["International"] = st.selectbox(
            "International Student", [1, 0],
            format_func=lambda v: "Yes" if v == 1 else "No",
            index=0 if intl_val == 1 else 1,
        )
        nac_labels = HUMAN_READABLE_LABELS["Nacionality"]
        curr_nac = int(current_data.get("Nacionality", 1))
        n_keys = list(nac_labels.keys())
        n_idx = n_keys.index(curr_nac) if curr_nac in n_keys else 0
        user_features["Nacionality"] = st.selectbox(
            "Nationality", n_keys,
            format_func=lambda k: nac_labels.get(k, str(k)),
            index=n_idx,
        )

with tab_socio:
    col1, col2 = st.columns(2)
    with col1:
        tuition_val = int(current_data.get("Tuition fees up to date", 1))
        user_features["Tuition fees up to date"] = st.selectbox(
            "Tuition Fees Up to Date", [1, 0],
            format_func=lambda v: "Yes (Current)" if v == 1 else "No (Overdue)",
            index=0 if tuition_val == 1 else 1,
        )
        debt_val = int(current_data.get("Debtor", 0))
        user_features["Debtor"] = st.selectbox(
            "Administrative Debtor Status", [1, 0],
            format_func=lambda v: "Yes (Has Unpaid Debt)" if v == 1 else "No (Clear)",
            index=0 if debt_val == 1 else 1,
        )
        schol_val = int(current_data.get("Scholarship holder", 0))
        user_features["Scholarship holder"] = st.selectbox(
            "Scholarship Recipient", [1, 0],
            format_func=lambda v: "Yes (Scholarship Awarded)" if v == 1 else "No",
            index=0 if schol_val == 1 else 1,
        )
        # Retain parent attributes from current data
        user_features["Mother's qualification"] = int(current_data.get("Mother's qualification", 1))
        user_features["Father's qualification"] = int(current_data.get("Father's qualification", 1))
        user_features["Mother's occupation"] = int(current_data.get("Mother's occupation", 5))
        user_features["Father's occupation"] = int(current_data.get("Father's occupation", 5))

    with col2:
        st.markdown("##### Regional Macroeconomic Context")
        st.caption("Annual macroeconomic rates at the time of cohort enrollment")
        user_features["Unemployment rate"] = st.number_input(
            "Unemployment Rate (%)", min_value=0.0, max_value=30.0,
            value=float(current_data.get("Unemployment rate", 11.1)), step=0.1,
        )
        user_features["Inflation rate"] = st.number_input(
            "Inflation Rate (%)", min_value=-5.0, max_value=20.0,
            value=float(current_data.get("Inflation rate", 1.4)), step=0.1,
        )
        user_features["GDP"] = st.number_input(
            "GDP Growth Rate (%)", min_value=-15.0, max_value=15.0,
            value=float(current_data.get("GDP", 1.74)), step=0.01,
        )

st.divider()

# Prediction Execution
st.markdown("### 3. Generate Prediction")

if st.button("🚀 Analyze Student Academic Trajectory", type="primary", use_container_width=True):
    try:
        with st.spinner(f"Computing prediction with {selected_model_name}..."):
            result = predict_student(user_features, model_path=selected_model_path)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Color-coded outcome display
            if predicted_class == "Graduate":
                st.success(f"### Predicted Outcome: **{predicted_class}**")
                st.caption("The student displays positive academic momentum consistent with degree completion.")
            elif predicted_class == "Dropout":
                st.error(f"### Predicted Outcome: **{predicted_class}** Risk")
                st.caption("Credit pace, evaluations, or administrative indicators align with dropout vulnerability.")
            else:
                st.info(f"### Predicted Outcome: **{predicted_class}** (Active / In-Progress)")
                st.caption("The student remains actively enrolled with an intermediate progression trajectory.")

            st.metric("Model Confidence Score", f"{confidence:.1%}")
            st.caption("Note: Represents model-derived softmax class score, not a calibrated real-world probability.")

        with res_col2:
            st.markdown("#### Outcome Probability Distribution")
            prob_df = pd.DataFrame({
                "Outcome": list(probabilities.keys()),
                "Probability": list(probabilities.values()),
            })
            st.bar_chart(prob_df.set_index("Outcome"), color="#008080", use_container_width=True)

        st.divider()

        # Supportive Academic Advising Guidance
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

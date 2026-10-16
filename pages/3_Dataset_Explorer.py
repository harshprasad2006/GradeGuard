"""
GradeGuard - Page 3: Dataset Explorer
Exploratory data analysis of the official UCI Machine Learning Repository Dataset ID 697.
Inspects shape, missingness, duplicates, distributions, class imbalance, and feature statistics.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.abspath("."))
from src.data_processing import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    BINARY_FEATURES,
    NUMERICAL_FEATURES,
    NOMINAL_CATEGORICAL_FEATURES,
    HUMAN_READABLE_LABELS,
)

st.set_page_config(page_title="Dataset Explorer - GradeGuard", page_icon="🔍", layout="wide")

st.title("🔍 UCI Dataset Explorer")
st.markdown("""
Interactive data exploration of the official **Predict Students' Dropout and Academic Success** dataset 
(UCI Dataset ID 697, DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)), 
curated by Realinho et al. (2021) from the Instituto Politécnico de Portalegre, Portugal.
""")

st.divider()

# Load Dataset
if not os.path.exists(RAW_DATA_PATH):
    st.error(f"Dataset file not found at {RAW_DATA_PATH}.")
    st.stop()

df = pd.read_csv(RAW_DATA_PATH)

# High-Level Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Records", f"{df.shape[0]:,}")
with col2:
    st.metric("Total Features", f"{df.shape[1] - 1}")
with col3:
    st.metric("Missing Values", f"{df.isnull().sum().sum()}")
with col4:
    st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
with col5:
    st.metric("Target Classes", f"{df[TARGET_COLUMN].nunique()}")

st.divider()

# Provenance and Missingness Explanation
st.markdown("### 1. Data Provenance & Why Incomplete Values are Absent")
st.info("""
**Archival Explanation (Realinho et al., 2021):**
Users often wonder why real-world student datasets have exactly zero missing values. 
As documented in the original publication (*Data* 2021, 6(11), 110):
1. **Institutional Cohort Curation:** The dataset aggregates records across 17 undergraduate programs between 2008/09 and 2018/19. Incomplete registrations or unconfirmed administrative enrolments were filtered at the database extraction phase in the Academic Management System (SIGA).
2. **Definitional Evaluation Scores:** For curricular units, students who did not sit for exams or attend coursework are formally recorded as receiving 0 evaluations and 0 grades, which represents a concrete academic outcome rather than an omitted value.
3. **Macroeconomic Determinism:** Macroeconomic variables (Unemployment, Inflation, GDP) from the Portuguese National Statistics Institute (INE/PORDATA) were deterministically mapped to each student's enrollment academic year, leaving no unmapped years.
""")

st.divider()

# Class Distribution & Imbalance
st.markdown("### 2. Class Proportions & Split Integrity")

col_dist1, col_dist2 = st.columns([1, 1.2])

with col_dist1:
    class_counts = df[TARGET_COLUMN].value_counts()
    class_props = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({
        "Outcome Class": class_counts.index,
        "Student Count": class_counts.values,
        "Proportion (%)": [round(p, 2) for p in class_props.values],
    })
    st.markdown("##### Target Class Breakdown")
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Class Imbalance Impact:**
    - `Graduate` is the dominant class (49.9%), while `Enrolled` is the clear minority (18.0%).
    - A naïve baseline classifier predicting only the majority class would achieve approximately **50% accuracy** while completely failing to detect any at-risk student.
    - This highlights why **Macro F1** (which averages performance unweighted across all three classes) is our primary evaluation metric.
    """)

with col_dist2:
    class_img = os.path.join("reports", "figures", "class_distribution.png")
    if os.path.exists(class_img):
        st.image(class_img, caption="Figure: Stratified Class Distribution Across Cohort Splits", use_container_width=True)

st.divider()

# Feature Taxonomy & Interactive Inspector
st.markdown("### 3. Feature Taxonomy & Interactive Inspector")

tax_col1, tax_col2, tax_col3 = st.columns(3)
with tax_col1:
    st.markdown(f"**Binary Features ({len(BINARY_FEATURES)}):**")
    st.caption(", ".join(BINARY_FEATURES))
with tax_col2:
    st.markdown(f"**Numerical Features ({len(NUMERICAL_FEATURES)}):**")
    st.caption(", ".join(NUMERICAL_FEATURES))
with tax_col3:
    st.markdown(f"**Nominal Categorical ({len(NOMINAL_CATEGORICAL_FEATURES)}):**")
    st.caption(", ".join(NOMINAL_CATEGORICAL_FEATURES))

st.markdown("##### Inspect Individual Feature Distributions")
selected_feature = st.selectbox("Select a feature to explore:", [c for c in df.columns if c != TARGET_COLUMN], index=24)

col_f1, col_f2 = st.columns([1, 1.2])

with col_f1:
    st.markdown(f"###### Descriptive Statistics for `{selected_feature}`")
    series = df[selected_feature]
    if pd.api.types.is_numeric_dtype(series):
        stats_df = pd.DataFrame({
            "Statistic": ["Count", "Mean", "Std Dev", "Min", "25th Percentile", "Median (50%)", "75th Percentile", "Max"],
            "Value": [
                f"{series.count():,}",
                f"{series.mean():.3f}",
                f"{series.std():.3f}",
                f"{series.min():.3f}",
                f"{series.quantile(0.25):.3f}",
                f"{series.median():.3f}",
                f"{series.quantile(0.75):.3f}",
                f"{series.max():.3f}",
            ],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.write(series.value_counts())

with col_f2:
    st.markdown(f"###### Distribution by Outcome Class (`{selected_feature}`)")
    if pd.api.types.is_numeric_dtype(series):
        # Grouped mean by target
        mean_by_target = df.groupby(TARGET_COLUMN)[selected_feature].mean().reset_index()
        st.bar_chart(mean_by_target.set_index(TARGET_COLUMN), color="#008080", use_container_width=True)
        st.caption(f"Average `{selected_feature}` across Graduate, Dropout, and Enrolled students.")
    else:
        st.bar_chart(df[selected_feature].value_counts(), color="#1B365D", use_container_width=True)

st.divider()

# Data Preview & Downloads
st.markdown("### 4. Cohort Data Preview")
st.caption("Displaying the first 25 records from the anonymized dataset.")
st.dataframe(df.head(25), use_container_width=True)

down_col1, down_col2 = st.columns(2)
with down_col1:
    with open(RAW_DATA_PATH, "rb") as f:
        st.download_button(
            label="Download Full Raw Dataset (CSV)",
            data=f,
            file_name="dataset_uci_697_raw.csv",
            mime="text/csv",
        )

with down_col2:
    if os.path.exists(TRAIN_DATA_PATH):
        with open(TRAIN_DATA_PATH, "rb") as f:
            st.download_button(
                label="Download Stratified Training Set (80%, CSV)",
                data=f,
                file_name="train_data_80pct.csv",
                mime="text/csv",
            )

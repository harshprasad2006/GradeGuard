"""
GradeGuard - Page 4: Methodology
Technical methodology, experimental protocol, and comprehensive literature review
comparing 5 peer-reviewed papers (2023-2026) alongside the official UCI dataset study.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath("."))

st.set_page_config(page_title="Methodology - GradeGuard", page_icon="📖", layout="wide")

st.title("📖 Technical Methodology & Literature Review")
st.markdown("""
This section outlines the mathematical framing, experimental protocol, data hygiene standards, 
and scholarly context underpinning **GradeGuard**.
""")

st.divider()

# 1. Literature Review Table
st.markdown("### 1. Peer-Reviewed Literature Review (2023–2026)")
st.markdown("""
To ground GradeGuard in state-of-the-art educational data mining (EDM), we synthesized findings from 
five verified peer-reviewed publications alongside the foundational UCI dataset paper (Realinho et al., 2021). 
*Note: Published findings are explicitly separated from the empirical experimental results of this project.*
""")

lit_data = [
    {
        "Citation": "Villar & de Andrade (2024)",
        "Title & Venue": "Supervised ML algorithms for predicting student dropout: a comparative study (Discover AI)",
        "DOI": "10.1007/s44163-023-00079-z",
        "Dataset & Context": "UCI 697 (Portugal, N=4,424)",
        "Models Evaluated": "LightGBM, CatBoost, XGBoost, RF, SVM, DT",
        "Key Findings": "Optuna-tuned gradient boosting frameworks outperformed traditional baselines; SMOTE effectively balanced minority classes.",
        "Limitations": "Relies on post-hoc synthetic oversampling which can introduce unrealistic feature interpolations in discrete curricular metrics.",
        "Relevance to GradeGuard": "Validates performance benchmarks on UCI 697; motivated our focus on comparing class-weighting inside pipelines.",
    },
    {
        "Citation": "Vaarma & Li (2024)",
        "Title & Venue": "Predicting student dropouts with ML: An empirical study in Finnish higher education (Tech in Society)",
        "DOI": "10.1016/j.techsoc.2024.102474",
        "Dataset & Context": "Finnish University LMS & Transcripts",
        "Models Evaluated": "Logistic Regression, Random Forest, XGBoost",
        "Key Findings": "Accumulated credits and failed courses were primary predictors; monthly LMS tracking added substantial temporal granularity.",
        "Limitations": "High dependency on Moodle/LMS telemetry which is unavailable in standard institutional tabular registries.",
        "Relevance to GradeGuard": "Affirms that credit accumulation rate dominates pre-entry demographics in explanatory power.",
    },
    {
        "Citation": "Christou et al. (2023)",
        "Title & Venue": "Performance and early drop prediction for higher education students using ML (Expert Syst Appl)",
        "DOI": "10.1016/j.eswa.2023.120079",
        "Dataset & Context": "Greek Higher Education cohort",
        "Models Evaluated": "SVM, Neural Networks, Decision Trees, KNN",
        "Key Findings": "Early drop-out detection accuracy improved substantially when 1st-year examination outcomes were incorporated.",
        "Limitations": "Evaluated binary drop/no-drop outcome, omitting the intermediate 'Enrolled' student classification.",
        "Relevance to GradeGuard": "Demonstrates the necessity of acknowledging the end-of-first-year measurement horizon rather than day-one claims.",
    },
    {
        "Citation": "Krüger, Britto & Barddal (2023)",
        "Title & Venue": "An explainable ML approach for student dropout prediction (Expert Syst Appl)",
        "DOI": "10.1016/j.eswa.2023.120933",
        "Dataset & Context": "Brazilian Multi-School Cohort (19 schools)",
        "Models Evaluated": "Random Forest, Extra Trees, XGBoost with SHAP/XAI",
        "Key Findings": "Explainable AI (XAI) is critical to uncover disparate drop reasons across educational stages and prevent algorithmic bias.",
        "Limitations": "Focused primarily on primary/secondary education where attendance laws differ from higher education.",
        "Relevance to GradeGuard": "Guides our implementation of supportive, non-punitive advisor guidance and feature importance breakdowns.",
    },
    {
        "Citation": "Cho, Yu & Kim (2023)",
        "Title & Venue": "A Study on Dropout Prediction for University Students Using ML (Applied Sciences)",
        "DOI": "10.3390/app132112004",
        "Dataset & Context": "South Korean University cohort",
        "Models Evaluated": "Logistic Regression, Random Forest, LightGBM",
        "Key Findings": "Class imbalance handling via cost-sensitive learning achieved higher recall on dropouts than threshold shifting.",
        "Limitations": "Institutional context specific to South Korean academic calendar and credit rules.",
        "Relevance to GradeGuard": "Directly supports our choice to utilize cost-sensitive class weighting (`class_weight='balanced'`) in classical pipelines.",
    },
    {
        "Citation": "Realinho et al. (2021) [Dataset Origin]",
        "Title & Venue": "Predicting Student Dropout and Academic Success (Data, 6(11), 110)",
        "DOI": "10.3390/data6110110",
        "Dataset & Context": "Instituto Politécnico de Portalegre (N=4,424)",
        "Models Evaluated": "Random Forest, Gradient Boosting, SVM",
        "Key Findings": "Established official 36-feature benchmark; 2nd semester approved units was single strongest predictor of completion.",
        "Limitations": "End-of-normal-duration data collection prevents real-time early intervention prior to first-year examinations.",
        "Relevance to GradeGuard": "Direct benchmark and data source for GradeGuard (UCI Repository ID 697).",
    },
]

lit_df = pd.DataFrame(lit_data)
st.dataframe(lit_df, use_container_width=True, hide_index=True)

st.divider()

# 2. Experimental Pipeline Protocol
st.markdown("### 2. Experimental Pipeline Architecture")

st.markdown("""
```mermaid
flowchart LR
    A["Raw UCI Dataset (N=4,424)"] --> B["Stratified 80/20 Split (Seed=42)"]
    B --> C["Train Set (N=3,539)"]
    B --> D["Held-Out Test Set (N=885)"]
    
    subgraph "Leak-Free Training Pipeline"
        C --> E["ColumnTransformer"]
        E --> F["Numerical: SimpleImputer + StandardScaler"]
        E --> G["Categorical: SimpleImputer + OneHotEncoder"]
        E --> H["Binary: SimpleImputer + Passthrough"]
        F & G & H --> I["5-Fold StratifiedKFold CV"]
        I --> J["GridSearchCV (Macro F1 Tuning)"]
        J --> K["Fitted Classical Classifiers"]
    end
    
    K --> L["Model Selection: Best Macro F1 (Random Forest)"]
    D --> M["Untouched Evaluation Pipeline"]
    L --> M
    M --> N["Held-Out Test Metrics & Diagnostic Figures"]
```
""")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown("#### Anti-Leakage Protocol")
    st.markdown("""
    - **Strict Stratification:** 80% train (3,539) / 20% test (885) partitioned using `random_state=42` stratified on `Target`.
    - **No Pre-Split Transformation:** Feature scalers, imputers, and encoders were fit strictly on training folds.
    - **Untouched Held-Out Test Set:** The test partition was locked and never inspected or tuned against until the final verification step.
    """)

with col_p2:
    st.markdown("#### Imbalance Mitigation Strategy")
    st.markdown("""
    - **Cost-Sensitive Learning First:** We evaluated `class_weight='balanced'` inside the cross-validation grid for all supporting models.
    - **Empirical Validation:** Cross-validation confirmed that balanced class weighting significantly improved minority `Enrolled` and `Dropout` recall without synthetic interpolation artifacts.
    - **Primary Metric:** Macro F1 was chosen as the tuning objective to weight all three outcomes equally.
    """)

"""
Extract and run diverse test cases from the held-out test set to verify model performance.
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.predict import predict_student, load_metadata
from src.data_processing import TARGET_COLUMN

SAMPLES_DIR = os.path.join("data", "test_cases")
os.makedirs(SAMPLES_DIR, exist_ok=True)

test_df = pd.read_csv("data/processed/test_data.csv")

# 5 distinct cases from held-out test set
# Case 1: Strong Academic Performer (Ground Truth: Graduate)
case1_df = test_df[
    (test_df[TARGET_COLUMN] == "Graduate") &
    (test_df["Curricular units 1st sem (approved)"] >= 5) &
    (test_df["Curricular units 2nd sem (approved)"] >= 5) &
    (test_df["Tuition fees up to date"] == 1)
].iloc[0:1]

# Case 2: Severe Academic & Financial Distress (Ground Truth: Dropout)
case2_df = test_df[
    (test_df[TARGET_COLUMN] == "Dropout") &
    (test_df["Curricular units 1st sem (approved)"] == 0) &
    (test_df["Curricular units 2nd sem (approved)"] == 0)
].iloc[0:1]

# Case 3: Active Intermediate Progression (Ground Truth: Enrolled)
case3_df = test_df[
    (test_df[TARGET_COLUMN] == "Enrolled") &
    (test_df["Curricular units 2nd sem (approved)"] >= 3)
].iloc[0:1]

# Case 4: Strong Sem 1 but Sudden Sem 2 Drop (Ground Truth: Dropout)
case4_df = test_df[
    (test_df[TARGET_COLUMN] == "Dropout") &
    (test_df["Curricular units 1st sem (approved)"] >= 4) &
    (test_df["Curricular units 2nd sem (approved)"] <= 1)
].iloc[0:1]

# Case 5: Mature / Working Student with High Attendance (Ground Truth: Graduate)
case5_df = test_df[
    (test_df[TARGET_COLUMN] == "Graduate") &
    (test_df["Age at enrollment"] >= 25)
].iloc[0:1]

test_cases = [
    ("Case_1_High_Performer_Graduate", "High Academic Performer (6/6 units approved, High GPA)", case1_df),
    ("Case_2_Severe_Dropout_Risk", "Severe Credit Deficit (0 units approved, Administrative flags)", case2_df),
    ("Case_3_Active_Enrolled_Student", "Active Moderate Progression (3-4 units approved, In Progress)", case3_df),
    ("Case_4_Sudden_Second_Sem_Slump", "Sem 1 Success but Sem 2 Collapse (Dropout Risk)", case4_df),
    ("Case_5_Mature_Student_Success", "Mature Student (Age 25+) with Good Completion", case5_df),
]

summary_results = []

print("=" * 80)
print("TEST CASE EXECUTION & VERIFICATION")
print("=" * 80)

for filename, desc, df_case in test_cases:
    true_label = df_case[TARGET_COLUMN].values[0]
    features_df = df_case.drop(columns=[TARGET_COLUMN])
    features_dict = features_df.iloc[0].to_dict()
    
    # Save 1-row CSV for direct Streamlit upload
    csv_file = os.path.join(SAMPLES_DIR, f"{filename}.csv")
    features_df.to_csv(csv_file, index=False)
    
    # Run prediction
    pred_res = predict_student(features_dict)
    predicted_label = pred_res["predicted_class"]
    conf = pred_res["confidence"]
    probs = pred_res["probabilities"]
    
    is_match = (true_label == predicted_label)
    match_str = "MATCH (CORRECT)" if is_match else "MISMATCH"
    
    print(f"\n[{filename}] - {desc}")
    print(f"  • Ground Truth Label:  {true_label}")
    print(f"  • Model Prediction:    {predicted_label} ({match_str})")
    print(f"  • Model Confidence:    {conf:.2%}")
    print(f"  • Class Probabilities: Dropout={probs['Dropout']:.2f}, Enrolled={probs['Enrolled']:.2f}, Graduate={probs['Graduate']:.2f}")
    print(f"  • Key Features: Age={features_dict.get('Age at enrollment')}, Sem1 Appr={features_dict.get('Curricular units 1st sem (approved)')}/{features_dict.get('Curricular units 1st sem (enrolled)')}, Sem2 Appr={features_dict.get('Curricular units 2nd sem (approved)')}/{features_dict.get('Curricular units 2nd sem (enrolled)')}, Tuition={features_dict.get('Tuition fees up to date')}, Debtor={features_dict.get('Debtor')}")
    print(f"  • Saved CSV to: {csv_file}")
    
    summary_results.append({
        "Case": filename,
        "Description": desc,
        "Ground_Truth": true_label,
        "Predicted": predicted_label,
        "Status": match_str,
        "Confidence": f"{conf:.1%}",
        "Prob_Dropout": probs["Dropout"],
        "Prob_Enrolled": probs["Enrolled"],
        "Prob_Graduate": probs["Graduate"],
        "CSV_Path": csv_file,
    })

print("\n" + "=" * 80)
print("TEST SUMMARY COMPLETE")
print("=" * 80)

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(SAMPLES_DIR, "test_cases_summary.csv"), index=False)

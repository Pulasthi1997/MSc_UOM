import os
import streamlit as st
import pandas as pd

from backend import train_model, load_artifacts, predict_by_msisdn, ARTIFACT_PATH

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📱", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size: 34px;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 6px;
        }
        .sub-title {
            font-size: 16px;
            color: #6b7280;
            margin-bottom: 24px;
        }
        .result-box {
            padding: 18px;
            border-radius: 14px;
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            margin-top: 10px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Customer Churn Prediction Interface</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter a phone number to view churn probability, prediction, and key reasons.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload behavioral Excel file", type=["xlsx"])

    if uploaded_file is not None:
        excel_path = "uploaded_behavioral_table.xlsx"
        with open(excel_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Excel uploaded successfully.")
    else:
        excel_path = "final_behavioural_table.xlsx"
        if os.path.exists(excel_path):
            st.info("Using local file: final_behavioural_table.xlsx")
        else:
            st.warning("Upload the Excel file or place final_behavioural_table.xlsx in this folder.")

    if st.button("Train / Refresh Model", use_container_width=True):
        if not os.path.exists(excel_path):
            st.error("Excel file not found.")
        else:
            with st.spinner("Training model..."):
                artifacts = train_model(excel_path)
            st.success("Model trained and saved.")
            st.write("Validation metrics:")
            st.json(artifacts["metrics"])

if os.path.exists(ARTIFACT_PATH):
    artifacts = load_artifacts()
else:
    artifacts = None

col1, col2 = st.columns([2, 1])

with col1:
    msisdn_input = st.text_input("Enter Phone Number (MSISDN)", placeholder="Example: 94740013413")

with col2:
    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn:
    if artifacts is None:
        st.error("Model is not trained yet. Please upload the Excel and click 'Train / Refresh Model' first.")
    elif not msisdn_input.strip():
        st.warning("Please enter a phone number.")
    else:
        result = predict_by_msisdn(msisdn_input, artifacts)

        if result is None:
            st.error("Phone number not found in the Excel data.")
        else:
            customer_row = result["customer_row"]
            probability = result["probability"]
            prediction = result["prediction"]
            risk_segment = result["risk_segment"]
            threshold = result["threshold"]
            reasons = result["reasons"]
            explanation_df = result["explanation_df"]

            prediction_label = "CHURN" if prediction == 1 else "NON-CHURN"

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Churn Probability", f"{probability:.2%}")
            c2.metric("Prediction", prediction_label)
            c3.metric("Risk Segment", risk_segment)
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Why this prediction?")
            for i, reason in enumerate(reasons, start=1):
                st.write(f"{i}. {reason}")

            st.subheader("Customer Details")
            show_cols = [c for c in ["MSISDN", "SERVICE_NAME", "MONTH_PRD"] if c in customer_row.columns]
            st.dataframe(customer_row[show_cols], use_container_width=True)

            st.subheader("Feature Values Used for Prediction")
            feature_cols = artifacts["features"]
            st.dataframe(customer_row[feature_cols], use_container_width=True)

            st.subheader("Top Feature Contributions")
            st.dataframe(
                explanation_df[["feature", "value", "shap_impact"]].head(10),
                use_container_width=True
            )

            st.caption(f"Decision threshold used: {threshold:.2f}")

st.markdown("---")
st.markdown("### Notes")
st.write(
    "This app predicts churn using the latest available customer record from the uploaded behavioral Excel file."
)
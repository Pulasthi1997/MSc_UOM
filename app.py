import streamlit as st
from backend import train_model_from_repo_data, predict_customer

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


@st.cache_resource
def load_artifacts():
    return train_model_from_repo_data()


artifacts = load_artifacts()

st.markdown('<div class="main-title">Customer Churn Prediction Interface</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter a phone number to view churn probability, prediction, and key reasons.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 1])

with col1:
    msisdn_input = st.text_input("Enter Phone Number (MSISDN)", placeholder="Example: 740013413")

with col2:
    st.write("")
    st.write("")
    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn:
    if not msisdn_input.strip():
        st.warning("Please enter a phone number.")
    else:
        result = predict_customer(msisdn_input, artifacts)

        if result is None:
            st.error("Phone number not found in the dataset.")
        else:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Churn Probability", f"{result['probability']:.2%}")
            c2.metric("Prediction", result["prediction"])
            c3.metric("Risk Segment", result["risk_segment"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("Why this prediction?")
            for i, reason in enumerate(result["reasons"], start=1):
                st.write(f"{i}. {reason}")

            st.subheader("Customer Details")
            show_cols = [c for c in ["MSISDN", "SERVICE_NAME", "MONTH_PRD"] if c in result["customer_row"].columns]
            if show_cols:
                st.dataframe(result["customer_row"][show_cols], use_container_width=True)

            st.subheader("Feature Values Used")
            feature_cols = artifacts["features"]
            st.dataframe(result["customer_row"][feature_cols], use_container_width=True)

            st.subheader("Top Feature Contributions")
            st.dataframe(
                result["explanation_df"][["feature", "value", "impact"]].head(10),
                use_container_width=True
            )

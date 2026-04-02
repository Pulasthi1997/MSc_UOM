import pandas as pd
import streamlit as st

from backend import (
    train_model_from_repo_data,
    get_services_for_msisdn,
    predict_customer,
    get_top_10_risky_customers,
    predict_batch,
    convert_df_to_csv
)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📱", layout="wide")

st.markdown("""
<style>
.badge-high {
    background-color: #fee2e2;
    color: #b91c1c;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.badge-medium {
    background-color: #fef3c7;
    color: #b45309;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.badge-low {
    background-color: #dcfce7;
    color: #15803d;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    return train_model_from_repo_data()


def risk_badge(risk):
    if risk == "HIGH RISK":
        return '<span class="badge-high">HIGH RISK</span>'
    elif risk == "MEDIUM RISK":
        return '<span class="badge-medium">MEDIUM RISK</span>'
    return '<span class="badge-low">LOW RISK</span>'


artifacts = load_artifacts()

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Top 10 Risky Customers", "Batch Upload"])

with tab1:
    st.caption("Enter phone number and select service to predict churn")

    msisdn_input = st.text_input(
        "Enter Phone Number (MSISDN)",
        placeholder="Example: 740013413"
    )

    services = []
    service_name = None

    if msisdn_input.strip():
        services = get_services_for_msisdn(msisdn_input, artifacts)

        if services:
            service_name = st.selectbox("Select Service Name", services)
        else:
            st.warning("No services found for this phone number.")

    predict_btn = st.button("Predict")

    if predict_btn:
        if not msisdn_input.strip():
            st.warning("Please enter a phone number.")
        elif not service_name:
            st.warning("Please select a valid service.")
        else:
            result = predict_customer(msisdn_input, service_name, artifacts)

            if result is None:
                st.error("Customer/service combination not found.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Churn Probability", f"{result['probability']:.2%}")
                c2.metric("Prediction", result["prediction"])
                c3.markdown(risk_badge(result["risk_segment"]), unsafe_allow_html=True)

                st.subheader("Why this prediction?")
                for i, reason in enumerate(result["reasons"], start=1):
                    st.write(f"{i}. {reason}")

                st.subheader("Customer Details")
                show_cols = [
                    c for c in ["MSISDN", "SERVICE_NAME", "MONTH_PRD"]
                    if c in result["customer_row"].columns
                ]
                if show_cols:
                    st.dataframe(result["customer_row"][show_cols], use_container_width=True)

                st.subheader("Feature Values Used")
                st.dataframe(
                    result["customer_row"][artifacts["features"]],
                    use_container_width=True
                )

                st.subheader("Top Feature Contributions")
                st.dataframe(
                    result["explanation_df"][["feature", "value", "impact"]].head(10),
                    use_container_width=True
                )

                csv_data = convert_df_to_csv(result["result_df"])
                safe_service = str(service_name).replace(" ", "_").replace("/", "_")
                st.download_button(
                    "Download Prediction Result as CSV",
                    data=csv_data,
                    file_name=f"prediction_{msisdn_input}_{safe_service}.csv",
                    mime="text/csv"
                )

with tab2:
    st.caption("Highest predicted churn risk customers")

    top10 = get_top_10_risky_customers(artifacts)
    show_cols = ["MSISDN", "SERVICE_NAME", "churn_probability", "prediction"]
    show_cols = [c for c in show_cols if c in top10.columns]

    st.dataframe(top10[show_cols], use_container_width=True)

    csv_top10 = convert_df_to_csv(top10[show_cols])
    st.download_button(
        "Download Top 10 Risky Customers",
        data=csv_top10,
        file_name="top_10_risky_customers.csv",
        mime="text/csv"
    )

with tab3:
    st.caption("Upload a CSV with MSISDN column. SERVICE_NAME column is optional.")

    batch_file = st.file_uploader("Upload CSV", type=["csv"])

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)

        st.write("Uploaded Data")
        st.dataframe(batch_df.head(), use_container_width=True)

        if "MSISDN" not in batch_df.columns:
            st.error("CSV must contain an `MSISDN` column.")
        else:
            batch_result = predict_batch(batch_df, artifacts)

            st.subheader("Batch Prediction Results")
            st.dataframe(batch_result, use_container_width=True)

            csv_batch = convert_df_to_csv(batch_result)
            st.download_button(
                "Download Batch Prediction Results",
                data=csv_batch,
                file_name="batch_prediction_results.csv",
                mime="text/csv"
            )

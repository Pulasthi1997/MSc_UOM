import pandas as pd
import streamlit as st

from backend import (
    train_model_from_repo_data,
    get_services_for_msisdn,
    get_all_services,
    predict_customer,
    predict_business_input,
    get_top_10_risky_customers,
    predict_batch,
    convert_df_to_csv,
    simulate_what_if
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
.badge-critical {
    background-color: #7f1d1d;
    color: white;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.badge-priority-high {
    background-color: #fca5a5;
    color: #7f1d1d;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.badge-priority-medium {
    background-color: #fde68a;
    color: #92400e;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.badge-priority-low {
    background-color: #bbf7d0;
    color: #166534;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.info-box {
    padding: 16px;
    border-radius: 12px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
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


def priority_badge(priority):
    if priority == "CRITICAL":
        return '<span class="badge-critical">CRITICAL</span>'
    elif priority == "HIGH":
        return '<span class="badge-priority-high">HIGH</span>'
    elif priority == "MEDIUM":
        return '<span class="badge-priority-medium">MEDIUM</span>'
    return '<span class="badge-priority-low">LOW</span>'


artifacts = load_artifacts()

st.title("Intelligent Customer Churn Analytics and Recommendation System")
st.caption("Predict churn risk, explain customer behavior, and support business retention decisions.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single Prediction",
    "Business Quick Prediction",
    "Top 10 Risky Customers",
    "Batch Upload",
    "Decision Intelligence"
])

with tab1:
    st.caption("Enter phone number and select service to predict churn")

    msisdn_input = st.text_input(
        "Enter Phone Number (MSISDN)",
        placeholder="Example: 740013413",
        key="single_msisdn"
    )

    services = []
    service_name = None

    if msisdn_input.strip():
        services = get_services_for_msisdn(msisdn_input, artifacts)
        if services:
            service_name = st.selectbox("Select Service Name", services, key="single_service")
        else:
            st.warning("No services found for this phone number.")

    predict_btn = st.button("Predict", key="single_predict_btn")

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
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Churn Probability", f"{result['probability']:.2%}")
                c2.metric("Prediction", result["prediction"])
                c3.markdown(risk_badge(result["risk_segment"]), unsafe_allow_html=True)
                c4.markdown(priority_badge(result["priority_level"]), unsafe_allow_html=True)

                st.subheader("Why this prediction?")
                for i, reason in enumerate(result["reasons"], start=1):
                    st.write(f"{i}. {reason}")

                st.subheader("Customer Details")
                show_cols = [c for c in ["MSISDN", "SERVICE_NAME", "MONTH_PRD"] if c in result["customer_row"].columns]
                if show_cols:
                    st.dataframe(result["customer_row"][show_cols], width="stretch")

                st.subheader("Feature Values Used")
                st.dataframe(result["customer_row"][artifacts["features"]], width="stretch")

                st.subheader("Top Feature Contributions")
                st.dataframe(
                    result["explanation_df"][["feature", "value", "impact"]].head(10),
                    width="stretch"
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
    st.caption("Business user can estimate churn using service, months stayed, and average monthly spend")

    all_services = get_all_services(artifacts)
    selected_service = st.selectbox("Select Service Name", all_services, key="business_service")
    months_stayed = st.number_input("Number of Months Stayed", min_value=1, max_value=120, value=6, step=1)
    avg_spend = st.number_input("Average Spent per Month", min_value=0.0, value=100.0, step=10.0)

    business_predict_btn = st.button("Run Business Prediction", key="business_predict_btn")

    if business_predict_btn:
        result = predict_business_input(
            service_name=selected_service,
            months_stayed=months_stayed,
            avg_spend_per_month=avg_spend,
            artifacts=artifacts
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{result['probability']:.2%}")
        c2.metric("Prediction", result["prediction"])
        c3.markdown(risk_badge(result["risk_segment"]), unsafe_allow_html=True)
        c4.markdown(priority_badge(result["priority_level"]), unsafe_allow_html=True)

        st.subheader("Business Recommendation")
        st.markdown(f"""
        <div class="info-box">
            <b>Recommended Action:</b> {result["recommended_action"]}<br>
            <b>Suggested Channel:</b> {result["suggested_channel"]}<br>
            <b>Business Reason:</b> {result["business_reason"]}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Derived Feature Values")
        st.dataframe(result["customer_row"], width="stretch")

        st.subheader("Top Feature Contributions")
        st.dataframe(
            result["explanation_df"][["feature", "value", "impact"]].head(10),
            width="stretch"
        )

        business_csv = convert_df_to_csv(result["result_df"])
        safe_service = str(selected_service).replace(" ", "_").replace("/", "_")
        st.download_button(
            "Download Business Prediction Result as CSV",
            data=business_csv,
            file_name=f"business_prediction_{safe_service}.csv",
            mime="text/csv"
        )

with tab3:
    st.caption("Highest predicted churn risk customers")

    top10 = get_top_10_risky_customers(artifacts)
    show_cols = [
        "MSISDN", "SERVICE_NAME", "churn_probability", "prediction",
        "risk_segment", "customer_value", "priority_score",
        "priority_level", "recommended_action"
    ]
    show_cols = [c for c in show_cols if c in top10.columns]

    st.dataframe(top10[show_cols], width="stretch")

    csv_top10 = convert_df_to_csv(top10[show_cols])
    st.download_button(
        "Download Top 10 Risky Customers",
        data=csv_top10,
        file_name="top_10_risky_customers.csv",
        mime="text/csv"
    )

with tab4:
    st.caption("Upload a CSV with MSISDN column. SERVICE_NAME column is optional.")

    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_file")

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)

        st.write("Uploaded Data")
        st.dataframe(batch_df.head(), width="stretch")

        if "MSISDN" not in batch_df.columns:
            st.error("CSV must contain an `MSISDN` column.")
        else:
            batch_result = predict_batch(batch_df, artifacts)

            st.subheader("Batch Prediction Results")
            st.dataframe(batch_result, width="stretch")

            csv_batch = convert_df_to_csv(batch_result)
            st.download_button(
                "Download Batch Prediction Results",
                data=csv_batch,
                file_name="batch_prediction_results.csv",
                mime="text/csv"
            )

with tab5:
    st.caption("AI-powered business decision support for retention strategy")

    msisdn_intel = st.text_input(
        "Enter Phone Number (MSISDN)",
        placeholder="Example: 740013413",
        key="intel_msisdn"
    )

    intel_services = []
    intel_service = None

    if msisdn_intel.strip():
        intel_services = get_services_for_msisdn(msisdn_intel, artifacts)
        if intel_services:
            intel_service = st.selectbox("Select Service Name", intel_services, key="intel_service")
        else:
            st.warning("No services found for this phone number.")

    run_intelligence = st.button("Run Decision Intelligence", key="run_intelligence")

    if run_intelligence:
        if not msisdn_intel.strip():
            st.warning("Please enter a phone number.")
        elif not intel_service:
            st.warning("Please select a valid service.")
        else:
            result = predict_customer(msisdn_intel, intel_service, artifacts)

            if result is None:
                st.error("Customer/service combination not found.")
            else:
                st.subheader("A. Next Best Action")
                st.markdown(f"""
                <div class="info-box">
                    <b>Recommended Action:</b> {result["recommended_action"]}<br>
                    <b>Suggested Channel:</b> {result["suggested_channel"]}<br>
                    <b>Business Reason:</b> {result["business_reason"]}
                </div>
                """, unsafe_allow_html=True)

                st.subheader("B. Priority Score")
                c1, c2, c3 = st.columns(3)
                c1.metric("Customer Value", f"{result['customer_value']:.2f}")
                c2.metric("Priority Score", f"{result['priority_score']:.2f}")
                c3.markdown(priority_badge(result["priority_level"]), unsafe_allow_html=True)

                st.subheader("C. What-if Simulator")
                selected_action = st.selectbox(
                    "Choose a simulated intervention",
                    ["Retention Discount", "Loyalty Reward", "Service Quality Improvement", "Bundle Upgrade"],
                    key="sim_action"
                )

                sim = simulate_what_if(
                    row_df=result["customer_row"],
                    model=artifacts["model"],
                    features=artifacts["features"],
                    threshold=artifacts["threshold"],
                    action_name=selected_action
                )

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Before Action", f"{sim['before_probability']:.2%}")
                sc2.metric("After Action", f"{sim['after_probability']:.2%}")
                sc3.metric("Change", f"{sim['probability_change']:.2%}")

                rc1, rc2 = st.columns(2)
                rc1.markdown(risk_badge(sim["before_risk"]), unsafe_allow_html=True)
                rc2.markdown(risk_badge(sim["after_risk"]), unsafe_allow_html=True)

                st.subheader("All Intelligence Summary")
                summary_df = pd.DataFrame([{
                    "MSISDN": msisdn_intel,
                    "SERVICE_NAME": intel_service,
                    "CURRENT_CHURN_PROBABILITY": result["probability"],
                    "CURRENT_RISK_SEGMENT": result["risk_segment"],
                    "RECOMMENDED_ACTION": result["recommended_action"],
                    "SUGGESTED_CHANNEL": result["suggested_channel"],
                    "CUSTOMER_VALUE": result["customer_value"],
                    "PRIORITY_SCORE": result["priority_score"],
                    "PRIORITY_LEVEL": result["priority_level"],
                    "SIMULATED_ACTION": selected_action,
                    "SIMULATED_CHURN_PROBABILITY": sim["after_probability"],
                    "SIMULATION_CHANGE": sim["probability_change"]
                }])

                st.dataframe(summary_df, width="stretch")

                intelligence_csv = convert_df_to_csv(summary_df)
                safe_service = str(intel_service).replace(" ", "_").replace("/", "_")
                st.download_button(
                    "Download Decision Intelligence Summary",
                    data=intelligence_csv,
                    file_name=f"decision_intelligence_{msisdn_intel}_{safe_service}.csv",
                    mime="text/csv"
                )

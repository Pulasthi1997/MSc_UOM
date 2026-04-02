import pandas as pd
import numpy as np
import shap

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

FEATURES = [
    "tot_amount_w_tax",
    "spend_lag_1",
    "spend_lag_2",
    "spend_lag_3",
    "spend_avg_last3",
    "spend_std_last3",
    "spend_trend_ratio",
    "consecutive_active_months",
    "service_risk_woe"
]

DATA_FILE = "final_behavioural_table.xlsx"


def load_data():
    df = pd.read_excel(DATA_FILE)

    required_cols = ["MSISDN", "SERVICE_NAME", "churn_flag"] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    df["MSISDN"] = df["MSISDN"].astype(str).str.strip()
    df["SERVICE_NAME"] = df["SERVICE_NAME"].astype(str).str.strip()

    if "MONTH_PRD" in df.columns:
        df["MONTH_PRD"] = pd.to_datetime(df["MONTH_PRD"], errors="coerce")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def prepare_latest_snapshot(df):
    if "MONTH_PRD" in df.columns:
        latest_df = (
            df.sort_values(["MSISDN", "SERVICE_NAME", "MONTH_PRD"])
            .groupby(["MSISDN", "SERVICE_NAME"], as_index=False)
            .tail(1)
            .copy()
        )
    else:
        latest_df = (
            df.drop_duplicates(subset=["MSISDN", "SERVICE_NAME"], keep="last")
            .copy()
        )
    return latest_df


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.50
    best_f1 = -1

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return float(best_threshold)


def classify_risk(prob, threshold):
    if prob >= threshold:
        return "HIGH RISK"
    elif prob >= threshold * 0.7:
        return "MEDIUM RISK"
    return "LOW RISK"


def compute_customer_value(row):
    current_spend = float(row.get("tot_amount_w_tax", 0) or 0)
    avg_spend = float(row.get("spend_avg_last3", 0) or 0)
    lag1 = float(row.get("spend_lag_1", 0) or 0)

    value = (0.5 * current_spend) + (0.3 * avg_spend) + (0.2 * lag1)
    return round(max(value, 0.0), 4)


def classify_priority(score):
    if score >= 70:
        return "CRITICAL"
    elif score >= 35:
        return "HIGH"
    elif score >= 15:
        return "MEDIUM"
    return "LOW"


def recommend_next_best_action(row, threshold):
    prob = float(row.get("churn_probability", 0))
    risk = classify_risk(prob, threshold)

    spend_trend_ratio = float(row.get("spend_trend_ratio", 0) or 0)
    active_months = float(row.get("consecutive_active_months", 0) or 0)
    service_risk_woe = float(row.get("service_risk_woe", 0) or 0)
    customer_value = compute_customer_value(row)

    if risk == "HIGH RISK" and customer_value >= 50 and spend_trend_ratio < 0.8:
        return {
            "recommended_action": "Premium retention offer",
            "suggested_channel": "Call Center + SMS",
            "business_reason": "High-value customer with strong churn risk and declining spend pattern."
        }

    if risk == "HIGH RISK" and active_months <= 3:
        return {
            "recommended_action": "Early-life engagement campaign",
            "suggested_channel": "SMS + App Push",
            "business_reason": "New or weakly engaged customer showing early churn signals."
        }

    if risk == "HIGH RISK" and service_risk_woe > 0:
        return {
            "recommended_action": "Service quality check and targeted retention package",
            "suggested_channel": "Call Center",
            "business_reason": "Service-level risk and behavioral indicators suggest proactive intervention."
        }

    if risk == "MEDIUM RISK" and customer_value >= 30:
        return {
            "recommended_action": "Personalized loyalty offer",
            "suggested_channel": "SMS",
            "business_reason": "Customer has moderate churn risk but meaningful commercial value."
        }

    if risk == "MEDIUM RISK":
        return {
            "recommended_action": "Reminder and engagement campaign",
            "suggested_channel": "SMS + Email",
            "business_reason": "Moderate churn indicators suggest low-cost retention action."
        }

    if customer_value >= 50:
        return {
            "recommended_action": "Upsell or bundle recommendation",
            "suggested_channel": "SMS + App Push",
            "business_reason": "Customer is stable and commercially valuable, making upsell more suitable than retention cost."
        }

    return {
        "recommended_action": "No immediate action",
        "suggested_channel": "None",
        "business_reason": "Customer currently shows low churn risk and low intervention need."
    }


def build_decision_intelligence_row(row, threshold):
    action_info = recommend_next_best_action(row, threshold)
    customer_value = compute_customer_value(row)
    priority_score = float(row.get("churn_probability", 0)) * customer_value

    return {
        "customer_value": customer_value,
        "priority_score": round(priority_score, 4),
        "priority_level": classify_priority(priority_score),
        "recommended_action": action_info["recommended_action"],
        "suggested_channel": action_info["suggested_channel"],
        "business_reason": action_info["business_reason"]
    }


def train_model_from_repo_data():
    df = load_data()

    train_df = df[df["churn_flag"].isin([0, 1])].copy()
    if train_df.empty:
        raise ValueError("No labeled rows found in churn_flag for training.")

    train_df["churn_flag"] = train_df["churn_flag"].astype(int)

    train_latest = prepare_latest_snapshot(train_df)

    X = train_latest[FEATURES].copy()
    y = train_latest["churn_flag"].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    class_weights = {0: 1.0, 1: neg / pos}

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
        class_weights=class_weights
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_prob)

    latest_snapshot = prepare_latest_snapshot(df).copy()
    latest_snapshot["churn_probability"] = model.predict_proba(latest_snapshot[FEATURES])[:, 1]
    latest_snapshot["prediction"] = np.where(
        latest_snapshot["churn_probability"] >= threshold,
        "CHURN",
        "NON-CHURN"
    )
    latest_snapshot["risk_segment"] = latest_snapshot["churn_probability"].apply(
        lambda x: classify_risk(x, threshold)
    )

    intelligence_cols = latest_snapshot.apply(
        lambda row: pd.Series(build_decision_intelligence_row(row, threshold)),
        axis=1
    )
    latest_snapshot = pd.concat([latest_snapshot, intelligence_cols], axis=1)
    latest_snapshot = latest_snapshot.loc[:, ~latest_snapshot.columns.duplicated()].copy()

    return {
        "model": model,
        "threshold": threshold,
        "latest_snapshot": latest_snapshot,
        "features": FEATURES
    }


def get_all_services(artifacts):
    df = artifacts["latest_snapshot"].copy()
    return sorted(df["SERVICE_NAME"].dropna().astype(str).unique().tolist())


def get_services_for_msisdn(msisdn, artifacts):
    df = artifacts["latest_snapshot"].copy()
    msisdn = str(msisdn).strip()

    services = (
        df[df["MSISDN"] == msisdn]["SERVICE_NAME"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return services


def get_service_default_woe(service_name, artifacts):
    df = artifacts["latest_snapshot"].copy()
    service_name = str(service_name).strip()

    service_rows = df[df["SERVICE_NAME"] == service_name].copy()

    if service_rows.empty:
        return 0.0

    if "service_risk_woe" in service_rows.columns:
        return float(service_rows["service_risk_woe"].median())

    return 0.0


def explain_prediction(model, row_df, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_df[features])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_row = shap_values[0]

    exp_df = pd.DataFrame({
        "feature": features,
        "value": row_df[features].iloc[0].values,
        "impact": shap_row
    })

    exp_df["abs_impact"] = exp_df["impact"].abs()
    exp_df = exp_df.sort_values("abs_impact", ascending=False)

    reasons = []
    for _, r in exp_df.head(3).iterrows():
        direction = "increased" if r["impact"] > 0 else "reduced"
        reasons.append(
            f"{r['feature']} = {round(float(r['value']), 4)} {direction} churn risk"
        )

    return exp_df, reasons


def simulate_what_if(row_df, model, features, threshold, action_name):
    sim_df = row_df.copy()

    if action_name == "Retention Discount":
        sim_df["tot_amount_w_tax"] = sim_df["tot_amount_w_tax"] * 1.10
        sim_df["spend_avg_last3"] = sim_df["spend_avg_last3"] * 1.08
        sim_df["spend_trend_ratio"] = sim_df["spend_trend_ratio"] * 1.15

    elif action_name == "Loyalty Reward":
        sim_df["consecutive_active_months"] = sim_df["consecutive_active_months"] + 1
        sim_df["spend_trend_ratio"] = sim_df["spend_trend_ratio"] * 1.05

    elif action_name == "Service Quality Improvement":
        sim_df["service_risk_woe"] = sim_df["service_risk_woe"] * 0.75
        sim_df["spend_trend_ratio"] = sim_df["spend_trend_ratio"] * 1.05

    elif action_name == "Bundle Upgrade":
        sim_df["tot_amount_w_tax"] = sim_df["tot_amount_w_tax"] * 1.12
        sim_df["spend_avg_last3"] = sim_df["spend_avg_last3"] * 1.10
        sim_df["consecutive_active_months"] = sim_df["consecutive_active_months"] + 1

    before_prob = float(model.predict_proba(row_df[features])[:, 1][0])
    after_prob = float(model.predict_proba(sim_df[features])[:, 1][0])

    before_risk = classify_risk(before_prob, threshold)
    after_risk = classify_risk(after_prob, threshold)

    delta = after_prob - before_prob

    return {
        "before_probability": before_prob,
        "after_probability": after_prob,
        "before_risk": before_risk,
        "after_risk": after_risk,
        "probability_change": delta,
        "simulated_row": sim_df
    }


def predict_customer(msisdn, service_name, artifacts):
    msisdn = str(msisdn).strip()
    service_name = str(service_name).strip()

    latest_snapshot = artifacts["latest_snapshot"].copy()

    row_df = latest_snapshot[
        (latest_snapshot["MSISDN"] == msisdn) &
        (latest_snapshot["SERVICE_NAME"] == service_name)
    ].copy()

    if row_df.empty:
        return None

    row_df = row_df.iloc[[0]].copy()
    X_input = row_df[artifacts["features"]].copy().fillna(0)

    model = artifacts["model"]
    threshold = artifacts["threshold"]

    prob = float(model.predict_proba(X_input)[:, 1][0])
    pred = int(prob >= threshold)
    risk = classify_risk(prob, threshold)

    exp_df, reasons = explain_prediction(model, row_df, artifacts["features"])

    intelligence = build_decision_intelligence_row(
        {**row_df.iloc[0].to_dict(), "churn_probability": prob},
        threshold
    )

    result_df = row_df.copy()
    result_df["churn_probability"] = prob
    result_df["prediction"] = "CHURN" if pred == 1 else "NON-CHURN"
    result_df["risk_segment"] = risk
    result_df["customer_value"] = intelligence["customer_value"]
    result_df["priority_score"] = intelligence["priority_score"]
    result_df["priority_level"] = intelligence["priority_level"]
    result_df["recommended_action"] = intelligence["recommended_action"]
    result_df["suggested_channel"] = intelligence["suggested_channel"]
    result_df["business_reason"] = intelligence["business_reason"]

    return {
        "probability": prob,
        "prediction": "CHURN" if pred == 1 else "NON-CHURN",
        "risk_segment": risk,
        "reasons": reasons,
        "customer_row": row_df,
        "explanation_df": exp_df,
        "result_df": result_df,
        "customer_value": intelligence["customer_value"],
        "priority_score": intelligence["priority_score"],
        "priority_level": intelligence["priority_level"],
        "recommended_action": intelligence["recommended_action"],
        "suggested_channel": intelligence["suggested_channel"],
        "business_reason": intelligence["business_reason"]
    }


def predict_business_input(service_name, months_stayed, avg_spend_per_month, artifacts):
    model = artifacts["model"]
    threshold = artifacts["threshold"]
    features = artifacts["features"]

    avg_spend_per_month = float(avg_spend_per_month)
    months_stayed = int(months_stayed)
    service_risk_woe = get_service_default_woe(service_name, artifacts)

    row_df = pd.DataFrame([{
        "SERVICE_NAME": service_name,
        "tot_amount_w_tax": avg_spend_per_month,
        "spend_lag_1": avg_spend_per_month,
        "spend_lag_2": avg_spend_per_month,
        "spend_lag_3": avg_spend_per_month,
        "spend_avg_last3": avg_spend_per_month,
        "spend_std_last3": 0.0,
        "spend_trend_ratio": 1.0,
        "consecutive_active_months": months_stayed,
        "service_risk_woe": service_risk_woe
    }])

    X_input = row_df[features].copy().fillna(0)

    prob = float(model.predict_proba(X_input)[:, 1][0])
    pred = int(prob >= threshold)
    risk = classify_risk(prob, threshold)

    exp_df, reasons = explain_prediction(model, row_df, features)

    intelligence = build_decision_intelligence_row(
        {**row_df.iloc[0].to_dict(), "churn_probability": prob},
        threshold
    )

    result_df = row_df.copy()
    result_df["churn_probability"] = prob
    result_df["prediction"] = "CHURN" if pred == 1 else "NON-CHURN"
    result_df["risk_segment"] = risk
    result_df["customer_value"] = intelligence["customer_value"]
    result_df["priority_score"] = intelligence["priority_score"]
    result_df["priority_level"] = intelligence["priority_level"]
    result_df["recommended_action"] = intelligence["recommended_action"]
    result_df["suggested_channel"] = intelligence["suggested_channel"]
    result_df["business_reason"] = intelligence["business_reason"]

    return {
        "probability": prob,
        "prediction": "CHURN" if pred == 1 else "NON-CHURN",
        "risk_segment": risk,
        "reasons": reasons,
        "customer_row": row_df,
        "explanation_df": exp_df,
        "result_df": result_df,
        "customer_value": intelligence["customer_value"],
        "priority_score": intelligence["priority_score"],
        "priority_level": intelligence["priority_level"],
        "recommended_action": intelligence["recommended_action"],
        "suggested_channel": intelligence["suggested_channel"],
        "business_reason": intelligence["business_reason"]
    }


def get_top_10_risky_customers(artifacts):
    df = artifacts["latest_snapshot"].copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df.sort_values("churn_probability", ascending=False).head(10).copy()


def predict_batch(msisdn_df, artifacts):
    results = []

    for _, r in msisdn_df.iterrows():
        msisdn = str(r["MSISDN"]).strip()

        service_name = None
        if "SERVICE_NAME" in msisdn_df.columns and pd.notna(r.get("SERVICE_NAME")):
            service_name = str(r["SERVICE_NAME"]).strip()

        available = get_services_for_msisdn(msisdn, artifacts)

        if not available:
            results.append({
                "MSISDN": msisdn,
                "SERVICE_NAME": service_name if service_name else "",
                "status": "NOT FOUND"
            })
            continue

        if not service_name or service_name not in available:
            service_name = available[0]

        pred = predict_customer(msisdn, service_name, artifacts)

        if pred is None:
            results.append({
                "MSISDN": msisdn,
                "SERVICE_NAME": service_name,
                "status": "NOT FOUND"
            })
        else:
            results.append({
                "MSISDN": msisdn,
                "SERVICE_NAME": service_name,
                "churn_probability": pred["probability"],
                "prediction": pred["prediction"],
                "risk_segment": pred["risk_segment"],
                "customer_value": pred["customer_value"],
                "priority_score": pred["priority_score"],
                "priority_level": pred["priority_level"],
                "recommended_action": pred["recommended_action"],
                "suggested_channel": pred["suggested_channel"],
                "status": "SUCCESS"
            })

    return pd.DataFrame(results)


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

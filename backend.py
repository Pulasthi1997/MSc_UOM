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

    df["MSISDN"] = df["MSISDN"].astype(str).str.strip()

    if "MONTH_PRD" in df.columns:
        df["MONTH_PRD"] = pd.to_datetime(df["MONTH_PRD"], errors="coerce")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def prepare_latest_snapshot(df):
    if "MONTH_PRD" in df.columns:
        latest_df = (
            df.sort_values(["MSISDN", "MONTH_PRD"])
              .groupby("MSISDN", as_index=False)
              .tail(1)
              .copy()
        )
    else:
        latest_df = df.drop_duplicates(subset=["MSISDN"], keep="last").copy()

    return latest_df


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = -1

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return float(best_threshold)


def train_model_from_repo_data():
    df = load_data()

    if "churn_flag" not in df.columns:
        raise ValueError("Excel file must contain churn_flag column.")

    train_df = df[df["churn_flag"].isin([0, 1])].copy()
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

    latest_snapshot = prepare_latest_snapshot(df)

    return {
        "model": model,
        "threshold": threshold,
        "latest_snapshot": latest_snapshot,
        "features": FEATURES
    }


def classify_risk(prob, threshold):
    if prob >= threshold:
        return "HIGH RISK"
    elif prob >= threshold * 0.7:
        return "MEDIUM RISK"
    return "LOW RISK"


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
        reasons.append(f"{r['feature']} = {round(float(r['value']), 4)} {direction} churn risk")

    return exp_df, reasons


def predict_customer(msisdn, artifacts):
    msisdn = str(msisdn).strip()

    latest_snapshot = artifacts["latest_snapshot"].copy()
    latest_snapshot["MSISDN"] = latest_snapshot["MSISDN"].astype(str).str.strip()

    row_df = latest_snapshot[latest_snapshot["MSISDN"] == msisdn].copy()

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

    return {
        "probability": prob,
        "prediction": "CHURN" if pred == 1 else "NON-CHURN",
        "risk_segment": risk,
        "reasons": reasons,
        "customer_row": row_df,
        "explanation_df": exp_df
    }

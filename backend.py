import os
import joblib
import numpy as np
import pandas as pd
import shap

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report

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

ARTIFACT_PATH = "churn_artifacts.pkl"


def load_behavioral_data(excel_file: str) -> pd.DataFrame:
    df = pd.read_excel(excel_file)

    required_cols = ["MSISDN"] + FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Excel: {missing_cols}")

    if "MONTH_PRD" in df.columns:
        df["MONTH_PRD"] = pd.to_datetime(df["MONTH_PRD"].astype(str), errors="coerce")

    df["MSISDN"] = df["MSISDN"].astype(str).str.strip()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURES] = df[FEATURES].fillna(0)

    return df


def prepare_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["MSISDN"]
    if "MONTH_PRD" in df.columns:
        sort_cols.append("MONTH_PRD")

    latest_df = (
        df.sort_values(sort_cols)
          .groupby("MSISDN", as_index=False)
          .tail(1)
          .copy()
    )

    return latest_df


def prepare_training_data(df: pd.DataFrame):
    if "churn_flag" not in df.columns:
        raise ValueError("Column 'churn_flag' not found in Excel file.")

    train_df = df[df["churn_flag"].isin([0, 1])].copy()

    if train_df.empty:
        raise ValueError("No labeled rows found. 'churn_flag' must contain 0/1 values.")

    train_df["churn_flag"] = train_df["churn_flag"].astype(int)

    latest_train_df = prepare_latest_snapshot(train_df)

    X = latest_train_df[FEATURES].copy()
    y = latest_train_df["churn_flag"].copy()

    return latest_train_df, X, y


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.50
    best_f1 = -1

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return float(best_threshold), float(best_f1)


def train_model(excel_file: str, artifact_path: str = ARTIFACT_PATH):
    df = load_behavioral_data(excel_file)
    latest_snapshot = prepare_latest_snapshot(df)
    train_df, X, y = prepare_training_data(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    class_weights = {0: 1.0, 1: neg / pos}

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        class_weights=class_weights
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_val, val_prob)

    roc_auc = roc_auc_score(y_val, val_prob)
    pr_auc = average_precision_score(y_val, val_prob)
    val_pred = (val_prob >= best_threshold).astype(int)
    report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)

    artifacts = {
        "model": model,
        "threshold": best_threshold,
        "features": FEATURES,
        "latest_snapshot": latest_snapshot,
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "best_f1": float(best_f1),
            "threshold": float(best_threshold),
            "precision_churn": float(report["1"]["precision"]) if "1" in report else 0.0,
            "recall_churn": float(report["1"]["recall"]) if "1" in report else 0.0
        }
    }

    joblib.dump(artifacts, artifact_path)
    return artifacts


def load_artifacts(artifact_path: str = ARTIFACT_PATH):
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Artifact file not found: {artifact_path}. Run training first."
        )
    return joblib.load(artifact_path)


def classify_risk(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "HIGH RISK"
    elif probability >= threshold * 0.70:
        return "MEDIUM RISK"
    return "LOW RISK"


def explain_prediction(model, row_df: pd.DataFrame, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_df[features])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if len(np.array(shap_values).shape) == 2:
        shap_row = shap_values[0]
    else:
        shap_row = shap_values

    explanation_df = pd.DataFrame({
        "feature": features,
        "value": row_df[features].iloc[0].values,
        "shap_impact": shap_row
    })

    explanation_df["abs_impact"] = explanation_df["shap_impact"].abs()
    explanation_df = explanation_df.sort_values("abs_impact", ascending=False)

    top_reasons = []
    for _, r in explanation_df.head(3).iterrows():
        direction = "increased" if r["shap_impact"] > 0 else "reduced"
        top_reasons.append(
            f"{r['feature']} = {round(float(r['value']), 4)} {direction} churn risk"
        )

    return explanation_df, top_reasons


def predict_by_msisdn(msisdn: str, artifacts: dict):
    model = artifacts["model"]
    threshold = artifacts["threshold"]
    features = artifacts["features"]
    latest_snapshot = artifacts["latest_snapshot"].copy()

    msisdn = str(msisdn).strip()
    latest_snapshot["MSISDN"] = latest_snapshot["MSISDN"].astype(str).str.strip()

    customer_df = latest_snapshot[latest_snapshot["MSISDN"] == msisdn].copy()

    if customer_df.empty:
        return None

    row_df = customer_df.iloc[[0]].copy()
    X_input = row_df[features].copy().fillna(0)

    probability = float(model.predict_proba(X_input)[:, 1][0])
    prediction = int(probability >= threshold)
    risk = classify_risk(probability, threshold)

    explanation_df, top_reasons = explain_prediction(model, row_df, features)

    return {
        "customer_row": row_df,
        "probability": probability,
        "prediction": prediction,
        "risk_segment": risk,
        "threshold": threshold,
        "reasons": top_reasons,
        "explanation_df": explanation_df
    }
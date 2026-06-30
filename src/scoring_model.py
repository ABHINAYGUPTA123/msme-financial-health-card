"""
scoring_model.py
------------------
Converts the feature store into a multidimensional MSME Financial Health
Score (0-100) made up of 5 explainable sub-scores, plus an ML-based
default-risk probability trained on the same features.

Sub-scores (each 0-100, higher = healthier):
  1. Revenue Stability   <- GST turnover level + low volatility + filing consistency
  2. Cash Flow Health    <- AA bank balance, inflow/outflow margin, low EMI bounces
  3. Digital Footprint   <- UPI transaction volume, ticket size, low bounce rate
  4. Workforce Stability <- EPFO employee count trend + compliance
  5. Compliance Score    <- GST + EPFO filing/payment consistency combined

Final composite score = weighted average of the 5 sub-scores.
Risk grade is bucketed from the composite score.

A simple Logistic Regression is also trained on synthetic labels (derived
from the same hidden health signal) to demonstrate how the system would
plug in real default/repayment outcome data once available — this gives
a P(default) alongside the explainable score, which is what credit teams
actually want: a human-readable score AND a calibrated risk probability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

WEIGHTS = {
    "revenue_stability": 0.25,
    "cash_flow_health": 0.25,
    "digital_footprint": 0.20,
    "workforce_stability": 0.15,
    "compliance_score": 0.15,
}


def _scale_0_100(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(50.0, index=s.index)
    scaled = (s - s.min()) / (s.max() - s.min()) * 100
    return 100 - scaled if invert else scaled


def compute_subscores(fs: pd.DataFrame) -> pd.DataFrame:
    df = fs.copy()

    df["revenue_stability"] = (
        0.5 * _scale_0_100(df["avg_monthly_turnover"])
        + 0.3 * _scale_0_100(df["turnover_volatility"], invert=True)
        + 0.2 * _scale_0_100(df["gst_filing_consistency"])
    )

    df["cash_flow_health"] = (
        0.4 * _scale_0_100(df["avg_bank_balance"])
        + 0.4 * _scale_0_100(df["cash_flow_margin"])
        + 0.2 * _scale_0_100(df["emi_bounce_rate"], invert=True)
    )

    df["digital_footprint"] = (
        0.5 * _scale_0_100(df["avg_monthly_txn_count"])
        + 0.2 * _scale_0_100(df["avg_ticket_size"])
        + 0.3 * _scale_0_100(df["avg_bounce_rate"], invert=True)
    )

    df["workforce_stability"] = (
        0.5 * _scale_0_100(df["avg_employee_count"])
        + 0.5 * _scale_0_100(df["employee_count_trend"])
    )

    df["compliance_score"] = (
        0.5 * _scale_0_100(df["gst_filing_consistency"])
        + 0.5 * _scale_0_100(df["epfo_compliance_rate"])
    )

    df["financial_health_score"] = sum(df[k] * w for k, w in WEIGHTS.items())
    df["financial_health_score"] = df["financial_health_score"].round(1)

    def grade(score):
        if score >= 75:
            return "A - Low Risk"
        elif score >= 55:
            return "B - Moderate Risk"
        elif score >= 35:
            return "C - High Risk"
        return "D - Very High Risk"

    df["risk_grade"] = df["financial_health_score"].apply(grade)
    return df


def train_default_risk_model(df: pd.DataFrame):
    """Train a tiny logistic regression as a stand-in ML risk model.
    In production this trains on actual loan outcome/repayment data fed
    back from the bank's LOS, via the AA/ULI feedback loop."""
    feature_cols = list(WEIGHTS.keys())
    X = df[feature_cols].values
    # synthetic label: lower composite score -> higher chance of "default"
    rng = np.random.default_rng(7)
    prob_default = 1 - (df["financial_health_score"] / 100)
    y = (rng.random(len(df)) < prob_default * 0.6).astype(int)

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(Xs, y)

    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, MODEL_DIR / "risk_model.pkl")
    df["pd_default_prob"] = model.predict_proba(Xs)[:, 1].round(3)
    return df, model, scaler


if __name__ == "__main__":
    from etl import build_feature_store

    fs = build_feature_store()
    scored = compute_subscores(fs)
    scored, model, scaler = train_default_risk_model(scored)
    out = Path(__file__).parent.parent / "data" / "scored_msmes.csv"
    scored.to_csv(out, index=False)
    print(f"Scored {len(scored)} MSMEs -> {out}")
    print(scored[["msme_id", "financial_health_score", "risk_grade", "pd_default_prob"]].head(10))

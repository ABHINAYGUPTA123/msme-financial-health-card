"""
app.py — MSME Financial Health Card dashboard
------------------------------------------------
Run locally:    streamlit run app.py
Deploy free:    push repo to GitHub -> share.streamlit.io -> New app
                (point it at this file, that's it — zero infra needed)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from etl import build_feature_store
from scoring_model import compute_subscores, train_default_risk_model, WEIGHTS

st.set_page_config(page_title="MSME Financial Health Card", layout="wide", page_icon="🏦")


@st.cache_data
def load_scored_data():
    fs = build_feature_store()
    scored = compute_subscores(fs)
    scored, _, _ = train_default_risk_model(scored)
    return scored


st.title("🏦 MSME Financial Health Card")
st.caption("AI/ML-driven multidimensional credit assessment using GST · UPI · Account Aggregator · EPFO data")

data = load_scored_data()

with st.sidebar:
    st.header("Select MSME Applicant")
    msme_id = st.selectbox("MSME ID", data["msme_id"].tolist())
    st.markdown("---")
    st.markdown("**Portfolio Snapshot**")
    st.metric("Total Applicants Scored", len(data))
    st.metric("Avg Health Score", round(data["financial_health_score"].mean(), 1))
    st.metric("Credit-worthy (Grade A/B)", f"{(data['risk_grade'].str[0].isin(['A','B'])).mean()*100:.0f}%")

row = data[data["msme_id"] == msme_id].iloc[0]

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("Financial Health Score", f"{row['financial_health_score']:.1f} / 100")
with col2:
    st.metric("Risk Grade", row["risk_grade"])
with col3:
    st.metric("Estimated Default Probability", f"{row['pd_default_prob']*100:.1f}%")

st.markdown("### Multidimensional Breakdown")
cols = st.columns(5)
labels = {
    "revenue_stability": "Revenue Stability",
    "cash_flow_health": "Cash Flow Health",
    "digital_footprint": "Digital Footprint",
    "workforce_stability": "Workforce Stability",
    "compliance_score": "Compliance Score",
}
for c, (key, label) in zip(cols, labels.items()):
    c.metric(label, f"{row[key]:.0f}", help=f"Weight in composite score: {WEIGHTS[key]*100:.0f}%")

# Radar chart
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=[row[k] for k in labels] + [row[list(labels.keys())[0]]],
    theta=list(labels.values()) + [list(labels.values())[0]],
    fill="toself",
    name=msme_id,
    line_color="#028090",
))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=False,
    height=420,
    margin=dict(l=40, r=40, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Raw Alt-Data Signals (driving the score)")
feature_cols = [c for c in data.columns if c not in
                 list(labels.keys()) + ["financial_health_score", "risk_grade", "pd_default_prob", "msme_id"]]
st.dataframe(row[["msme_id"] + feature_cols].to_frame().T, use_container_width=True)

st.markdown("### Full Portfolio")
st.dataframe(
    data[["msme_id", "financial_health_score", "risk_grade", "pd_default_prob"]].sort_values(
        "financial_health_score", ascending=False
    ),
    use_container_width=True,
    height=300,
)

st.markdown("---")
st.caption(
    "Prototype for IDBI Innovate 2026 · Track 03 — Financial Health Score · "
    "Team Bounty Hunter · Data is synthetic, generated to demonstrate the pipeline; "
    "production version connects to real GSTN / UPI switch / Setu-Sahamati AA / EPFO employer APIs."
)

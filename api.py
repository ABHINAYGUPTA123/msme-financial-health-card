import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from etl import build_feature_store
from scoring_model import compute_subscores, train_default_risk_model, WEIGHTS, _scale_0_100

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="MSME Financial Health Card | IDBI Bank",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CUSTOM CSS ────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #020f14 0%, #03252e 40%, #012a35 70%, #041e27 100%);
    color: #e2f4f7;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #011820 0%, #022535 100%);
    border-right: 1px solid #1a4a57;
}
[data-testid="stSidebar"] * { color: #c8eaf0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #7ecfdf !important; font-weight: 600; }

/* ── Top header bar ── */
.top-header {
    background: linear-gradient(90deg, #028090 0%, #014f5e 50%, #023040 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    border: 1px solid #036475;
    box-shadow: 0 8px 32px rgba(2,128,144,0.3);
}
.top-header h1 {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}
.top-header p {
    color: #a0dce6;
    font-size: 13px;
    margin: 4px 0 0 0;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #7ff0ff;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(127,240,255,0.3);
}

/* ── Score card ── */
.score-card {
    background: linear-gradient(135deg, #014f5e 0%, #012a35 100%);
    border: 1px solid #036475;
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform 0.2s;
}
.score-card:hover { transform: translateY(-3px); }
.score-card .label {
    color: #7ecfdf;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.score-card .value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
    line-height: 1;
}
.score-card .sub {
    color: #a0dce6;
    font-size: 11px;
    margin-top: 6px;
}

/* ── Sub-score cards ── */
.sub-card {
    background: linear-gradient(135deg, #012535 0%, #011820 100%);
    border: 1px solid #1a4a57;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    margin-bottom: 4px;
}
.sub-card .sub-label {
    color: #7ecfdf;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.sub-card .sub-val {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin: 4px 0;
}
.sub-card .weight-tag {
    color: #4db8c8;
    font-size: 10px;
}

/* ── Progress bar custom ── */
.prog-wrap { margin: 3px 0 8px 0; }
.prog-bg {
    background: #0a3040;
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    border: 1px solid #1a4a57;
}
.prog-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #028090, #02c39a);
    transition: width 0.5s ease;
}

/* ── Grade badge ── */
.grade-A { color: #00e5a0; font-weight: 800; font-size: 28px; }
.grade-B { color: #ffd166; font-weight: 800; font-size: 28px; }
.grade-C { color: #f4845f; font-weight: 800; font-size: 28px; }
.grade-D { color: #ef233c; font-weight: 800; font-size: 28px; }

/* ── Section heading ── */
.section-head {
    color: #7ecfdf;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-left: 3px solid #028090;
    padding-left: 10px;
    margin: 24px 0 14px 0;
}

/* ── Table styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a4a57 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Input form ── */
.input-card {
    background: linear-gradient(135deg, #012535 0%, #011820 100%);
    border: 1px solid #1a4a57;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.input-card h4 {
    color: #7ecfdf;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 0 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1a4a57;
}

/* ── Streamlit widget override ── */
.stSlider > div > div > div > div { background: #028090 !important; }
.stNumberInput input, .stTextInput input {
    background: #0a2535 !important;
    border: 1px solid #1a4a57 !important;
    color: #e2f4f7 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(90deg, #028090, #014f5e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 16px rgba(2,128,144,0.4) !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #02a0b5, #028090) !important;
    box-shadow: 0 6px 20px rgba(2,128,144,0.5) !important;
    transform: translateY(-1px);
}
.stTabs [data-baseweb="tab-list"] {
    background: #011820 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #1a4a57 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #7ecfdf !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #028090, #014f5e) !important;
    color: white !important;
}

/* ── Metric override ── */
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #7ecfdf !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #2a6070;
    font-size: 11px;
    padding: 20px 0 10px 0;
    border-top: 1px solid #1a4a57;
    margin-top: 32px;
}

/* ── Alert/info box ── */
.info-box {
    background: rgba(2,128,144,0.1);
    border: 1px solid #028090;
    border-radius: 10px;
    padding: 14px 18px;
    color: #a0dce6;
    font-size: 13px;
    margin: 12px 0;
}

/* hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── DATA LOAD ─────────────────────────────
@st.cache_data
def load_data():
    fs = build_feature_store()
    scored = compute_subscores(fs)
    scored, model, scaler = train_default_risk_model(scored)
    return scored, model, scaler

data, ml_model, ml_scaler = load_data()

LABELS = {
    "revenue_stability":   ("Revenue Stability",   "GST"),
    "cash_flow_health":    ("Cash Flow Health",     "AA"),
    "digital_footprint":   ("Digital Footprint",    "UPI"),
    "workforce_stability": ("Workforce Stability",  "EPFO"),
    "compliance_score":    ("Compliance Score",     "GST+EPFO"),
}

def grade_color(grade):
    g = grade[0]
    return {"A":"grade-A","B":"grade-B","C":"grade-C","D":"grade-D"}.get(g,"grade-B")

def score_color(s):
    if s >= 75: return "#00e5a0"
    if s >= 55: return "#ffd166"
    if s >= 35: return "#f4845f"
    return "#ef233c"

# ─────────────────────────── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 20px 0;'>
        <div style='font-size:36px;'>🏦</div>
        <div style='font-size:15px; font-weight:700; color:#ffffff; letter-spacing:-0.3px;'>IDBI Bank</div>
        <div style='font-size:11px; color:#4db8c8; letter-spacing:1px; text-transform:uppercase;'>MSME Credit Intelligence</div>
    </div>
    <hr style='border-color:#1a4a57; margin:0 0 20px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#7ecfdf; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🔍  Applicant Health Card", "✏️  Try Your Own Data", "📊  Portfolio Dashboard"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a4a57; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7ecfdf; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:12px;'>Portfolio Snapshot</div>", unsafe_allow_html=True)

    total   = len(data)
    avg_s   = data["financial_health_score"].mean()
    creditw = (data["risk_grade"].str[0].isin(["A","B"])).mean()*100
    npa_risk= (data["risk_grade"].str[0].isin(["D"])).mean()*100

    col1s, col2s = st.columns(2)
    col1s.metric("Total MSMEs", total)
    col2s.metric("Avg Score", f"{avg_s:.1f}")
    col1s.metric("Credit-worthy", f"{creditw:.0f}%")
    col2s.metric("High Risk", f"{npa_risk:.0f}%")

    st.markdown("<hr style='border-color:#1a4a57; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10px; color:#2a6070; text-align:center; line-height:1.6;'>
        Powered by GST · UPI · AA · EPFO<br>
        IDBI Innovate 2026 · Team Bounty Hunter
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — APPLICANT HEALTH CARD
# ═══════════════════════════════════════════════════════════════════
if "Applicant" in page:

    st.markdown("""
    <div class='top-header'>
        <h1>🏦 MSME Financial Health Card</h1>
        <p>AI/ML-driven multidimensional credit assessment using alternate data</p>
        <div style='margin-top:10px;'>
            <span class='badge'>GST</span>
            <span class='badge'>UPI</span>
            <span class='badge'>Account Aggregator</span>
            <span class='badge'>EPFO</span>
            <span class='badge'>IDBI Innovate 2026</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Applicant selector
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        msme_id = st.selectbox("Select MSME Applicant", data["msme_id"].tolist(),
                               help="Choose from 200 scored MSME applicants")
    with col_info:
        st.markdown("<div class='info-box'>📌 Select any MSME ID to view their complete Financial Health Card — score, risk grade, 5 sub-dimensions, and radar breakdown.</div>", unsafe_allow_html=True)

    row = data[data["msme_id"] == msme_id].iloc[0]
    sc  = row["financial_health_score"]
    grd = row["risk_grade"]
    pd_prob = row["pd_default_prob"]*100

    st.markdown("<div class='section-head'>Overall Assessment</div>", unsafe_allow_html=True)

    # 3 main metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='score-card'>
            <div class='label'>Financial Health Score</div>
            <div class='value' style='color:{score_color(sc)};'>{sc:.1f}<span style='font-size:16px; color:#7ecfdf;'>/100</span></div>
            <div class='sub'>Composite of 5 dimensions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='score-card'>
            <div class='label'>Risk Grade</div>
            <div class='{grade_color(grd)}'>{grd[0]}</div>
            <div class='sub'>{grd[4:]}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        pd_color = "#00e5a0" if pd_prob < 25 else "#ffd166" if pd_prob < 40 else "#ef233c"
        st.markdown(f"""
        <div class='score-card'>
            <div class='label'>Default Probability</div>
            <div class='value' style='color:{pd_color};'>{pd_prob:.1f}<span style='font-size:16px; color:#7ecfdf;'>%</span></div>
            <div class='sub'>ML model estimate</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        max_loan = int(row["avg_monthly_turnover"] * 3 / 100000)
        st.markdown(f"""
        <div class='score-card'>
            <div class='label'>Indicative Loan Limit</div>
            <div class='value' style='font-size:24px; color:#7ecfdf;'>₹{max_loan}L</div>
            <div class='sub'>3× avg monthly turnover</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sub-scores + Radar side by side
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='section-head'>5-Dimension Breakdown</div>", unsafe_allow_html=True)
        for key, (label, source) in LABELS.items():
            val = row[key]
            w   = WEIGHTS[key]*100
            bar_color = "#00e5a0" if val >= 70 else "#ffd166" if val >= 45 else "#f4845f"
            st.markdown(f"""
            <div class='sub-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div class='sub-label'>{label}</div>
                        <div class='weight-tag'>Source: {source} · Weight: {w:.0f}%</div>
                    </div>
                    <div class='sub-val' style='color:{bar_color};'>{val:.0f}</div>
                </div>
                <div class='prog-wrap'>
                    <div class='prog-bg'>
                        <div class='prog-fill' style='width:{val}%; background: linear-gradient(90deg,{bar_color}88,{bar_color});'></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-head'>Health Radar</div>", unsafe_allow_html=True)
        cats = [v[0] for v in LABELS.values()]
        vals = [row[k] for k in LABELS] + [row[list(LABELS.keys())[0]]]
        cats_closed = cats + [cats[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats_closed, fill="toself",
            line=dict(color="#02c39a", width=2.5),
            fillcolor="rgba(2,195,154,0.15)", name=msme_id,
        ))
        # Benchmark (avg)
        avg_vals = [data[k].mean() for k in LABELS] + [data[list(LABELS.keys())[0]].mean()]
        fig.add_trace(go.Scatterpolar(
            r=avg_vals, theta=cats_closed, fill="toself",
            line=dict(color="#7ecfdf", width=1.5, dash="dot"),
            fillcolor="rgba(126,207,223,0.05)", name="Portfolio Avg",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(1,24,32,0.6)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#1a4a57", tickfont=dict(color="#4db8c8", size=9), color="#4db8c8"),
                angularaxis=dict(gridcolor="#1a4a57", tickfont=dict(color="#7ecfdf", size=11)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#7ecfdf", size=11), bgcolor="rgba(0,0,0,0)"),
            height=400,
            margin=dict(l=40,r=40,t=20,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw signals table
    st.markdown("<div class='section-head'>Raw Alt-Data Signals</div>", unsafe_allow_html=True)
    feature_cols = [c for c in data.columns if c not in list(LABELS.keys()) + ["financial_health_score","risk_grade","pd_default_prob","msme_id"]]
    raw_df = row[feature_cols].to_frame().T.reset_index(drop=True)
    raw_df.columns = [c.replace("_"," ").title() for c in raw_df.columns]
    st.dataframe(raw_df.style.format("{:.2f}"), use_container_width=True, height=80)

    st.markdown("<div class='footer'>IDBI Innovate 2026 · Track 03 — Financial Health Score · Team Bounty Hunter · Prototype uses synthetic data</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 2 — TRY YOUR OWN DATA
# ═══════════════════════════════════════════════════════════════════
elif "Try" in page:

    st.markdown("""
    <div class='top-header'>
        <h1>✏️ Try Your Own MSME Data</h1>
        <p>Enter real or estimated figures to instantly compute your Financial Health Score</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='info-box'>💡 Fill in your business's approximate monthly figures. All inputs are estimates — the system computes your score in real time.</div>", unsafe_allow_html=True)

    with st.form("msme_form"):
        st.markdown("<br>", unsafe_allow_html=True)

        # GST
        st.markdown("<div class='input-card'><h4>📋 GST Data</h4>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        avg_turnover = g1.number_input("Avg Monthly GST Turnover (₹)", min_value=0, value=500000, step=50000, format="%d", help="Your average monthly sales turnover from GSTR-1/3B")
        gst_filing   = g2.slider("GST Filing Consistency (%)", 0, 100, 75, help="% of months you filed GST on time in last 12 months") / 100
        turnover_vol = g1.number_input("Turnover Volatility (₹ std dev)", min_value=0, value=80000, step=10000, format="%d", help="How much your turnover fluctuates month to month")
        st.markdown("</div>", unsafe_allow_html=True)

        # UPI
        st.markdown("<div class='input-card'><h4>📱 UPI Transaction Data</h4>", unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        upi_txns   = u1.number_input("Avg Monthly UPI Transactions", min_value=0, value=300, step=10)
        upi_ticket = u2.number_input("Avg Transaction Value (₹)", min_value=0, value=1500, step=100)
        upi_bounce = u3.slider("UPI Bounce/Failure Rate (%)", 0, 50, 5, help="% of UPI transactions that fail or bounce") / 100
        st.markdown("</div>", unsafe_allow_html=True)

        # AA
        st.markdown("<div class='input-card'><h4>🏦 Bank Account (Account Aggregator)</h4>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        avg_balance  = a1.number_input("Avg Bank Balance (₹)", min_value=0, value=200000, step=10000, format="%d")
        avg_inflow   = a2.number_input("Avg Monthly Inflow (₹)", min_value=1, value=600000, step=10000, format="%d")
        avg_outflow  = a3.number_input("Avg Monthly Outflow (₹)", min_value=0, value=450000, step=10000, format="%d")
        emi_bounce   = st.slider("EMI Bounce Rate (% of months)", 0, 100, 5, help="% of months where an EMI/loan payment bounced") / 100
        st.markdown("</div>", unsafe_allow_html=True)

        # EPFO
        st.markdown("<div class='input-card'><h4>👷 EPFO / Workforce Data</h4>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        emp_count  = e1.number_input("Current Employee Count", min_value=0, value=8, step=1)
        emp_trend  = e2.number_input("Employee Change (last 12 months)", min_value=-50, value=2, step=1, help="Positive = hired more, Negative = reduced workforce")
        epfo_comp  = e3.slider("EPFO Payment Compliance (%)", 0, 100, 80, help="% of months EPFO contributions paid on time") / 100
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🚀  Calculate My Financial Health Score")

    if submitted:
        # Build feature row manually
        cash_flow_margin = (avg_inflow - avg_outflow) / max(avg_inflow, 1)

        # We need to scale against the existing dataset range
        ref = data.copy()

        def score_single(val, col, invert=False):
            mn, mx = ref[col].min(), ref[col].max()
            if mx == mn: return 50.0
            s = (val - mn)/(mx - mn)*100
            return 100-s if invert else s

        rev_stab = (
            0.5 * score_single(avg_turnover,    "avg_monthly_turnover") +
            0.3 * score_single(turnover_vol,    "turnover_volatility", invert=True) +
            0.2 * score_single(gst_filing,      "gst_filing_consistency")
        )
        cash_hlth = (
            0.4 * score_single(avg_balance,     "avg_bank_balance") +
            0.4 * min(max(cash_flow_margin*100, 0), 100) +
            0.2 * score_single(emi_bounce,      "emi_bounce_rate", invert=True)
        )
        dig_foot = (
            0.5 * score_single(upi_txns,        "avg_monthly_txn_count") +
            0.2 * score_single(upi_ticket,      "avg_ticket_size") +
            0.3 * score_single(upi_bounce,      "avg_bounce_rate", invert=True)
        )
        work_stab = (
            0.5 * score_single(emp_count,       "avg_employee_count") +
            0.5 * score_single(emp_trend,       "employee_count_trend")
        )
        comp_sc = (
            0.5 * score_single(gst_filing,      "gst_filing_consistency") +
            0.5 * score_single(epfo_comp,       "epfo_compliance_rate")
        )

        sub_scores = {
            "revenue_stability":   rev_stab,
            "cash_flow_health":    cash_hlth,
            "digital_footprint":   dig_foot,
            "workforce_stability": work_stab,
            "compliance_score":    comp_sc,
        }
        final_score = sum(sub_scores[k]*w for k,w in WEIGHTS.items())
        final_score = min(max(final_score, 0), 100)

        if final_score >= 75:   grade = "A - Low Risk"
        elif final_score >= 55: grade = "B - Moderate Risk"
        elif final_score >= 35: grade = "C - High Risk"
        else:                   grade = "D - Very High Risk"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-head'>Your Financial Health Card</div>", unsafe_allow_html=True)

        sc_color = score_color(final_score)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='score-card'>
                <div class='label'>Your Health Score</div>
                <div class='value' style='color:{sc_color};'>{final_score:.1f}<span style='font-size:16px;color:#7ecfdf;'>/100</span></div>
                <div class='sub'>Based on your inputs</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='score-card'>
                <div class='label'>Risk Grade</div>
                <div class='{grade_color(grade)}'>{grade[0]}</div>
                <div class='sub'>{grade[4:]}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            pct = int((data["financial_health_score"] < final_score).mean()*100)
            st.markdown(f"""<div class='score-card'>
                <div class='label'>Better Than</div>
                <div class='value' style='color:#7ecfdf;'>{pct}<span style='font-size:16px;'>%</span></div>
                <div class='sub'>of all applicants in portfolio</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left2, right2 = st.columns([1,1])

        with left2:
            st.markdown("<div class='section-head'>Your Sub-Score Breakdown</div>", unsafe_allow_html=True)
            for key,(label,source) in LABELS.items():
                val = sub_scores[key]
                bar_color = "#00e5a0" if val>=70 else "#ffd166" if val>=45 else "#f4845f"
                st.markdown(f"""<div class='sub-card'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div><div class='sub-label'>{label}</div><div class='weight-tag'>Source: {source}</div></div>
                        <div class='sub-val' style='color:{bar_color};'>{val:.0f}</div>
                    </div>
                    <div class='prog-wrap'><div class='prog-bg'>
                        <div class='prog-fill' style='width:{val}%;background:linear-gradient(90deg,{bar_color}88,{bar_color});'></div>
                    </div></div>
                </div>""", unsafe_allow_html=True)

        with right2:
            st.markdown("<div class='section-head'>How You Compare</div>", unsafe_allow_html=True)
            cats  = [v[0] for v in LABELS.values()]
            yours = [sub_scores[k] for k in LABELS]
            avgs  = [data[k].mean() for k in LABELS]
            cats_c  = cats + [cats[0]]
            yours_c = yours + [yours[0]]
            avgs_c  = avgs  + [avgs[0]]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=yours_c, theta=cats_c, fill="toself",
                line=dict(color="#02c39a",width=2.5), fillcolor="rgba(2,195,154,0.2)", name="Your Score"))
            fig2.add_trace(go.Scatterpolar(r=avgs_c, theta=cats_c, fill="toself",
                line=dict(color="#7ecfdf",width=1.5,dash="dot"), fillcolor="rgba(126,207,223,0.05)", name="Portfolio Avg"))
            fig2.update_layout(
                polar=dict(
                    bgcolor="rgba(1,24,32,0.6)",
                    radialaxis=dict(visible=True,range=[0,100],gridcolor="#1a4a57",tickfont=dict(color="#4db8c8",size=9),color="#4db8c8"),
                    angularaxis=dict(gridcolor="#1a4a57",tickfont=dict(color="#7ecfdf",size=11)),
                ),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True, legend=dict(font=dict(color="#7ecfdf",size=11),bgcolor="rgba(0,0,0,0)"),
                height=380, margin=dict(l=40,r=40,t=20,b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='footer'>IDBI Innovate 2026 · Track 03 — Financial Health Score · Team Bounty Hunter</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 3 — PORTFOLIO DASHBOARD
# ═══════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class='top-header'>
        <h1>📊 Portfolio Dashboard</h1>
        <p>Bank-level view of all scored MSME applicants</p>
    </div>
    """, unsafe_allow_html=True)

    # Grade distribution bar
    st.markdown("<div class='section-head'>Risk Grade Distribution</div>", unsafe_allow_html=True)
    grade_counts = data["risk_grade"].value_counts().reset_index()
    grade_counts.columns = ["Grade","Count"]
    grade_counts["Color"] = grade_counts["Grade"].map({
        "A - Low Risk":"#00e5a0","B - Moderate Risk":"#ffd166",
        "C - High Risk":"#f4845f","D - Very High Risk":"#ef233c"
    })
    fig3 = go.Figure()
    for _, r in grade_counts.iterrows():
        fig3.add_trace(go.Bar(x=[r["Grade"]], y=[r["Count"]], marker_color=r["Color"],
                              name=r["Grade"], text=[r["Count"]], textposition="outside",
                              textfont=dict(color="#ffffff",size=14)))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(1,24,32,0.6)",
        xaxis=dict(gridcolor="#1a4a57",tickfont=dict(color="#7ecfdf")),
        yaxis=dict(gridcolor="#1a4a57",tickfont=dict(color="#7ecfdf")),
        showlegend=False, height=320, margin=dict(l=20,r=20,t=30,b=20),
        bargap=0.35,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Score distribution histogram
    col_h, col_s = st.columns(2)
    with col_h:
        st.markdown("<div class='section-head'>Score Distribution</div>", unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=data["financial_health_score"], nbinsx=20,
            marker=dict(color="#028090", line=dict(color="#02c39a",width=1)),
            opacity=0.85,
        ))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(1,24,32,0.6)",
            xaxis=dict(title="Health Score", gridcolor="#1a4a57", tickfont=dict(color="#7ecfdf"), titlefont=dict(color="#7ecfdf")),
            yaxis=dict(title="Count", gridcolor="#1a4a57", tickfont=dict(color="#7ecfdf"), titlefont=dict(color="#7ecfdf")),
            height=300, margin=dict(l=20,r=20,t=20,b=40), showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_s:
        st.markdown("<div class='section-head'>Score vs Default Probability</div>", unsafe_allow_html=True)
        fig5 = go.Figure()
        colors = data["risk_grade"].map({"A - Low Risk":"#00e5a0","B - Moderate Risk":"#ffd166","C - High Risk":"#f4845f","D - Very High Risk":"#ef233c"})
        fig5.add_trace(go.Scatter(
            x=data["financial_health_score"], y=data["pd_default_prob"]*100,
            mode="markers", marker=dict(color=colors, size=6, opacity=0.7),
        ))
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(1,24,32,0.6)",
            xaxis=dict(title="Health Score", gridcolor="#1a4a57", tickfont=dict(color="#7ecfdf"), titlefont=dict(color="#7ecfdf")),
            yaxis=dict(title="Default Prob %", gridcolor="#1a4a57", tickfont=dict(color="#7ecfdf"), titlefont=dict(color="#7ecfdf")),
            height=300, margin=dict(l=20,r=20,t=20,b=40), showlegend=False,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Full portfolio table
    st.markdown("<div class='section-head'>Full Applicant Portfolio</div>", unsafe_allow_html=True)
    display = data[["msme_id","financial_health_score","risk_grade","pd_default_prob","revenue_stability","cash_flow_health","digital_footprint"]].copy()
    display["pd_default_prob"] = (display["pd_default_prob"]*100).round(1)
    display.columns = ["MSME ID","Health Score","Risk Grade","Default % ","Revenue","Cash Flow","Digital"]
    display = display.sort_values("Health Score", ascending=False).reset_index(drop=True)
    st.dataframe(
        display.style
        .background_gradient(subset=["Health Score"], cmap="YlGn")
        .format({"Health Score":"{:.1f}","Default % ":"{:.1f}%","Revenue":"{:.0f}","Cash Flow":"{:.0f}","Digital":"{:.0f}"}),
        use_container_width=True, height=450,
    )

    st.markdown("<div class='footer'>IDBI Innovate 2026 · Track 03 — Financial Health Score · Team Bounty Hunter</div>", unsafe_allow_html=True)

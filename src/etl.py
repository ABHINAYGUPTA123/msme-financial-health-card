"""
etl.py — Data Engineering layer
--------------------------------
EXTRACT  -> read raw CSVs (stand-ins for GST/UPI/AA/EPFO API pulls)
TRANSFORM-> clean, validate, aggregate monthly -> per-MSME features
LOAD     -> write a single tidy "feature store" table
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path

# Works both locally and on Streamlit Cloud
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


def _ensure_raw_data():
    """Auto-generate synthetic data if raw CSVs don't exist yet."""
    DATA_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / "raw_gst.csv").exists():
        # Find generate_synthetic_data.py (root or data/ folder)
        gen_script = ROOT_DIR / "generate_synthetic_data.py"
        if not gen_script.exists():
            gen_script = ROOT_DIR / "data" / "generate_synthetic_data.py"
        subprocess.run([sys.executable, str(gen_script)], check=True)


def extract(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"raw_{name}.csv", parse_dates=["month"])
    return df


def transform_gst(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("msme_id").agg(
        avg_monthly_turnover=("gst_turnover", "mean"),
        turnover_volatility=("gst_turnover", "std"),
        gst_filing_consistency=("filed_on_time", "mean"),
    )
    g["turnover_volatility"] = g["turnover_volatility"].fillna(0)
    return g


def transform_upi(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("msme_id").agg(
        avg_monthly_txn_count=("upi_txn_count", "mean"),
        avg_ticket_size=("upi_avg_ticket", "mean"),
        avg_bounce_rate=("upi_bounce_rate", "mean"),
    )
    return g


def transform_aa(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("msme_id").agg(
        avg_bank_balance=("avg_bank_balance", "mean"),
        avg_inflow=("inflow", "mean"),
        avg_outflow=("outflow", "mean"),
        emi_bounce_rate=("emi_bounced", "mean"),
    )
    g["cash_flow_margin"] = (g["avg_inflow"] - g["avg_outflow"]) / g["avg_inflow"].replace(0, 1)
    return g


def transform_epfo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("msme_id").agg(
        avg_employee_count=("employee_count", "mean"),
        employee_count_trend=("employee_count", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0),
        epfo_compliance_rate=("epfo_paid_on_time", "mean"),
    )
    return g


def build_feature_store() -> pd.DataFrame:
    """Auto-generate data if needed, then Extract + Transform + Load."""
    _ensure_raw_data()   # <-- yeh line fix hai: pehle data banao

    gst  = transform_gst(extract("gst"))
    upi  = transform_upi(extract("upi"))
    aa   = transform_aa(extract("aa"))
    epfo = transform_epfo(extract("epfo"))

    feature_store = gst.join([upi, aa, epfo], how="inner")
    feature_store = feature_store.fillna(0).reset_index()
    return feature_store


if __name__ == "__main__":
    fs = build_feature_store()
    out_path = DATA_DIR / "feature_store.csv"
    fs.to_csv(out_path, index=False)
    print(f"Feature store built: {fs.shape[0]} MSMEs x {fs.shape[1]-1} features -> {out_path}")
    print(fs.head())

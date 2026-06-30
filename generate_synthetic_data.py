"""
generate_synthetic_data.py
---------------------------------
Simulates the 4 alternate-data sources an MSME Financial Health Card
would normally pull from real APIs:

  1. GST    -> monthly sales filings (GSTR-1/3B)
  2. UPI    -> merchant transaction stream
  3. AA     -> Account Aggregator bank statement summary
  4. EPFO   -> payroll / employee contribution records

In production each of these would be a real connector (GSTN API, UPI
switch, Setu/Sahamati AA SDK, EPFO employer API). For the prototype we
generate realistic-looking data so the rest of the pipeline (ETL ->
features -> scoring -> dashboard) can be demoed and deployed end-to-end
without needing bank-grade API access.

Run:
    python data/generate_synthetic_data.py
Produces:
    data/raw_gst.csv, raw_upi.csv, raw_aa.csv, raw_epfo.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT = Path(__file__).parent

N_MSME = 200            # number of MSME applicants
MONTHS = 12             # 12 months of trailing history

msme_ids = [f"MSME{1000+i}" for i in range(N_MSME)]

# Give each MSME a hidden "true health" between 0-1 that drives all its
# downstream numbers consistently (so the data isn't pure noise).
true_health = np.random.beta(2, 2, N_MSME)


def month_range():
    return pd.date_range(end=pd.Timestamp.today(), periods=MONTHS, freq="ME")


# ---------------------------------------------------------------- GST ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base_turnover = np.random.uniform(2, 50) * 1e5        # ₹2L–₹50L/month base
    for m in month_range():
        seasonal = 1 + 0.15 * np.sin(m.month)
        turnover = base_turnover * seasonal * (0.7 + 0.6 * h) * np.random.uniform(0.85, 1.15)
        filed_on_time = np.random.rand() < (0.5 + 0.5 * h)   # healthier MSMEs file on time more
        rows.append([mid, m.date(), round(turnover, 2), filed_on_time])
pd.DataFrame(rows, columns=["msme_id", "month", "gst_turnover", "filed_on_time"]).to_csv(
    OUT / "raw_gst.csv", index=False
)

# ---------------------------------------------------------------- UPI ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base_txn = np.random.uniform(50, 2000)
    for m in month_range():
        n_txns = max(5, int(base_txn * (0.6 + 0.8 * h) * np.random.uniform(0.8, 1.2)))
        avg_ticket = np.random.uniform(200, 5000)
        bounce_rate = max(0, np.random.normal(0.08 * (1 - h), 0.03))
        rows.append([mid, m.date(), n_txns, round(avg_ticket, 2), round(min(bounce_rate, 0.5), 4)])
pd.DataFrame(rows, columns=["msme_id", "month", "upi_txn_count", "upi_avg_ticket", "upi_bounce_rate"]).to_csv(
    OUT / "raw_upi.csv", index=False
)

# ----------------------------------------------------------------- AA ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base_balance = np.random.uniform(0.5, 20) * 1e5
    for m in month_range():
        avg_balance = base_balance * (0.6 + 0.8 * h) * np.random.uniform(0.85, 1.15)
        inflow = avg_balance * np.random.uniform(0.8, 1.5)
        outflow = inflow * (0.7 + 0.35 * (1 - h))
        bounced_emi = np.random.rand() < (0.25 * (1 - h))
        rows.append([mid, m.date(), round(avg_balance, 2), round(inflow, 2), round(outflow, 2), bounced_emi])
pd.DataFrame(
    rows, columns=["msme_id", "month", "avg_bank_balance", "inflow", "outflow", "emi_bounced"]
).to_csv(OUT / "raw_aa.csv", index=False)

# --------------------------------------------------------------- EPFO ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base_emp = max(1, int(np.random.uniform(1, 40) * (0.5 + h)))
    for m in month_range():
        employees = max(1, base_emp + np.random.randint(-1, 2))
        contribution_paid_on_time = np.random.rand() < (0.5 + 0.5 * h)
        rows.append([mid, m.date(), employees, contribution_paid_on_time])
pd.DataFrame(rows, columns=["msme_id", "month", "employee_count", "epfo_paid_on_time"]).to_csv(
    OUT / "raw_epfo.csv", index=False
)

print("Synthetic data generated in", OUT)
print("MSME count:", N_MSME, "| months/MSME:", MONTHS)

"""
generate_synthetic_data.py
--------------------------
Simulates 4 alternate-data sources: GST, UPI, AA, EPFO.
Writes raw_*.csv files into a 'data/' subfolder next to this script.
Called automatically by etl.py if raw files are missing.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Works from any directory — always writes to data/ folder
ROOT_DIR = Path(__file__).parent
OUT = ROOT_DIR / "data"
OUT.mkdir(exist_ok=True)

N_MSME  = 200
MONTHS  = 12

msme_ids    = [f"MSME{1000+i}" for i in range(N_MSME)]
true_health = np.random.beta(2, 2, N_MSME)


def month_range():
    return pd.date_range(end=pd.Timestamp.today(), periods=MONTHS, freq="ME")


# ---- GST ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base = np.random.uniform(2, 50) * 1e5
    for m in month_range():
        seasonal = 1 + 0.15 * np.sin(m.month)
        turnover = base * seasonal * (0.7 + 0.6*h) * np.random.uniform(0.85, 1.15)
        on_time  = np.random.rand() < (0.5 + 0.5*h)
        rows.append([mid, m.date(), round(turnover, 2), on_time])
pd.DataFrame(rows, columns=["msme_id","month","gst_turnover","filed_on_time"]).to_csv(OUT/"raw_gst.csv", index=False)

# ---- UPI ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base = np.random.uniform(50, 2000)
    for m in month_range():
        n      = max(5, int(base*(0.6+0.8*h)*np.random.uniform(0.8,1.2)))
        ticket = np.random.uniform(200, 5000)
        bounce = max(0, np.random.normal(0.08*(1-h), 0.03))
        rows.append([mid, m.date(), n, round(ticket,2), round(min(bounce,0.5),4)])
pd.DataFrame(rows, columns=["msme_id","month","upi_txn_count","upi_avg_ticket","upi_bounce_rate"]).to_csv(OUT/"raw_upi.csv", index=False)

# ---- AA ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base = np.random.uniform(0.5, 20) * 1e5
    for m in month_range():
        bal    = base*(0.6+0.8*h)*np.random.uniform(0.85,1.15)
        inflow = bal*np.random.uniform(0.8,1.5)
        outflow= inflow*(0.7+0.35*(1-h))
        bounce = np.random.rand() < (0.25*(1-h))
        rows.append([mid, m.date(), round(bal,2), round(inflow,2), round(outflow,2), bounce])
pd.DataFrame(rows, columns=["msme_id","month","avg_bank_balance","inflow","outflow","emi_bounced"]).to_csv(OUT/"raw_aa.csv", index=False)

# ---- EPFO ----
rows = []
for mid, h in zip(msme_ids, true_health):
    base = max(1, int(np.random.uniform(1,40)*(0.5+h)))
    for m in month_range():
        emp     = max(1, base+np.random.randint(-1,2))
        on_time = np.random.rand() < (0.5+0.5*h)
        rows.append([mid, m.date(), emp, on_time])
pd.DataFrame(rows, columns=["msme_id","month","employee_count","epfo_paid_on_time"]).to_csv(OUT/"raw_epfo.csv", index=False)

print(f"Synthetic data written to: {OUT}")

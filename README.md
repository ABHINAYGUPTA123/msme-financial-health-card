# MSME Financial Health Card
**IDBI Innovate 2026 · Track 03 — Financial Health Score**
Team: **Bounty Hunter** | Lead: Abhinay Gupta

AI/ML-driven MSME Financial Health Card that aggregates alternate data
(GST, UPI, Account Aggregator, EPFO) into a single multidimensional
credit score — built for **easy review and one-click deployment**.

## What's inside
```
msme-health-score/
├── data/
│   └── generate_synthetic_data.py   # simulates GST/UPI/AA/EPFO feeds
├── src/
│   ├── etl.py                       # Extract-Transform-Load -> feature store
│   └── scoring_model.py             # 5-dimension score + ML risk model
├── app.py                           # Streamlit dashboard (the "Health Card")
├── api.py                           # FastAPI scoring endpoint (ULI/OCEN hook)
└── requirements.txt
```

## Run locally (2 commands)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Dashboard opens at `http://localhost:8501`.

Optional — run the integration API in a second terminal:
```bash
uvicorn api:app --reload --port 8000
# POST http://localhost:8000/score   {"msme_id": "MSME1000"}
```

## Deploy in under 5 minutes (no infra needed)
1. Push this folder to a public GitHub repo.
2. **Dashboard:** go to [share.streamlit.io](https://share.streamlit.io) → "New app" →
   select the repo → main file `app.py` → Deploy. You get a public URL instantly.
3. **API (optional, for ULI/OCEN demo):** go to [render.com](https://render.com) →
   "New Web Service" → same repo → start command:
   `uvicorn api:app --host 0.0.0.0 --port $PORT`.

No Docker, no servers to manage, no cost — ideal for a hackathon submission
that needs a live deployment link by the deadline.

## How the scoring works
1. **ETL (`src/etl.py`)** ingests 4 alt-data sources and aggregates 12
   months of history per MSME into ~14 raw features.
2. **Scoring (`src/scoring_model.py`)** converts raw features into 5
   explainable sub-scores (Revenue Stability, Cash Flow Health, Digital
   Footprint, Workforce Stability, Compliance) and combines them into a
   weighted **Financial Health Score (0–100)**.
3. A small **Logistic Regression** model also produces a calibrated
   default-probability — designed to be retrained on real loan-outcome
   data once IDBI's repayment history is fed back in.
4. The **API layer** exposes `/score` as the integration point that a
   real ULI/OCEN/AA-connected LOS would call in real time.

## Swapping synthetic data for real data
Replace the 4 functions in `data/generate_synthetic_data.py` with real
connectors (GSTN API, UPI switch settlement files, Setu/Sahamati AA
SDK, EPFO employer portal) that write to the same `raw_*.csv` schema —
the ETL and scoring layers need no changes.

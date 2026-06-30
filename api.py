"""
api.py — Real-time scoring API (ULI / OCEN integration point)
-----------------------------------------------------------------
This is what IDBI's Loan Origination System (LOS) or the ULI/OCEN
network would actually call to get a near real-time Financial Health
Score for an MSME during loan processing.

Run locally:   uvicorn api:app --reload --port 8000
Try it:        POST http://localhost:8000/score   {"msme_id": "MSME1000"}
Deploy free:   Render.com / Railway.app -> "uvicorn api:app --host 0.0.0.0 --port $PORT"
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from etl import build_feature_store
from scoring_model import compute_subscores, train_default_risk_model

app = FastAPI(title="MSME Financial Health Score API", version="1.0")

_cache = {"data": None}


def get_scored_data():
    if _cache["data"] is None:
        fs = build_feature_store()
        scored = compute_subscores(fs)
        scored, _, _ = train_default_risk_model(scored)
        _cache["data"] = scored
    return _cache["data"]


class ScoreRequest(BaseModel):
    msme_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    data = get_scored_data()
    row = data[data["msme_id"] == req.msme_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="MSME ID not found in feature store")
    r = row.iloc[0]
    return {
        "msme_id": r["msme_id"],
        "financial_health_score": float(r["financial_health_score"]),
        "risk_grade": r["risk_grade"],
        "estimated_default_probability": float(r["pd_default_prob"]),
        "sub_scores": {
            "revenue_stability": float(r["revenue_stability"]),
            "cash_flow_health": float(r["cash_flow_health"]),
            "digital_footprint": float(r["digital_footprint"]),
            "workforce_stability": float(r["workforce_stability"]),
            "compliance_score": float(r["compliance_score"]),
        },
    }


@app.get("/portfolio")
def portfolio():
    data = get_scored_data()
    return data[["msme_id", "financial_health_score", "risk_grade"]].to_dict(orient="records")

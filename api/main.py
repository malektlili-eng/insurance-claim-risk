"""
FastAPI service for the Building Insurance Claim-Risk model.
================================================================================

Run with:

    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs.

Endpoints
---------
GET  /health          -> liveness, model metadata, data provenance, calibration
GET  /schema          -> accepted feature schema, categorical levels, risk bands
POST /predict         -> score a single building
POST /predict/batch   -> score a list of buildings + portfolio summary

Both scoring endpoints call `TrainedModel.score_frame`, the same method used by
train.py, the Streamlit app and the notebook (verified by tests/test_parity.py).
"""
from __future__ import annotations

import os
import sys
from typing import List, Literal, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src_pipeline import TrainedModel, BAND_NAMES  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")

app = FastAPI(
    title="Insurance Claim-Risk API",
    version="2.0.0",
    description=(
        "Scores an insured building and returns a calibrated claim probability, "
        "a decision at the business-tuned threshold, a risk band and a "
        "recommended underwriting action. Check `/health` → `data_source`: "
        "when it is `generated`, the model was trained on a reproducible "
        "synthetic dataset, not a real portfolio."
    ),
)

_model: Optional[TrainedModel] = None


def get_model() -> TrainedModel:
    global _model
    if _model is None:
        if not os.path.exists(os.path.join(MODEL_DIR, "model.joblib")):
            raise HTTPException(status_code=503,
                                detail="Model not trained yet. Run `python train.py` first.")
        _model = TrainedModel.load(MODEL_DIR)
    return _model


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class Building(BaseModel):
    """One insured building. Field names mirror the raw dataset columns.

    `Building Dimension`, `Garden` and `Geo_Code` may be null: the pipeline
    imputes them exactly as it does for missing values in training data.
    """
    YearOfObservation: int = Field(..., ge=2000, le=2100, examples=[2014])
    Insured_Period: float = Field(..., gt=0, le=1, examples=[1.0])
    Residential: int = Field(..., ge=0, le=1, examples=[1])
    Building_Painted: Literal["N", "V"] = Field(..., examples=["V"])
    Building_Fenced: Literal["N", "V"] = Field(..., examples=["N"])
    Garden: Optional[Literal["V", "O"]] = Field(None, examples=["V"])
    Settlement: Literal["U", "R"] = Field(..., examples=["U"])
    Building_Dimension: Optional[float] = Field(None, alias="Building Dimension",
                                                gt=0, examples=[1200.0])
    Building_Type: Literal["Fire-resistive", "Non-combustible",
                           "Ordinary", "Wood-framed"] = Field(..., examples=["Ordinary"])
    NumberOfWindows: Literal["without", "1", "2", "3", "4", "5", "6", "7",
                             "8", "9", ">=10"] = Field(..., examples=["4"])
    Geo_Code: Optional[Union[str, int]] = Field(None, examples=["0042"])

    model_config = {"populate_by_name": True}

    def to_raw(self) -> dict:
        return {
            "YearOfObservation": self.YearOfObservation,
            "Insured_Period": self.Insured_Period,
            "Residential": self.Residential,
            "Building_Painted": self.Building_Painted,
            "Building_Fenced": self.Building_Fenced,
            "Garden": self.Garden,
            "Settlement": self.Settlement,
            "Building Dimension": self.Building_Dimension,
            "Building_Type": self.Building_Type,
            "NumberOfWindows": self.NumberOfWindows,
            "Geo_Code": None if self.Geo_Code is None else str(self.Geo_Code),
        }


class Prediction(BaseModel):
    claim_probability: float
    prediction: Literal["oui", "non"]
    risk_level: Literal["Faible", "Moyen", "Élevé"]
    recommended_action: str
    threshold_used: float


class BatchRequest(BaseModel):
    buildings: List[Building]


class BatchResponse(BaseModel):
    count: int
    results: List[Prediction]
    summary: dict


def _score(buildings: List[Building]) -> List[Prediction]:
    m = get_model()
    raw = pd.DataFrame([b.to_raw() for b in buildings])
    scored = m.score_frame(raw)
    return [Prediction(claim_probability=float(r.claim_probability),
                       prediction=r.prediction, risk_level=r.risk_level,
                       recommended_action=r.recommended_action,
                       threshold_used=m.threshold)
            for r in scored.itertuples(index=False)]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    try:
        m = get_model()
    except HTTPException:
        return {"status": "ok", "model_trained": False}
    cal = m.metrics.get("calibration", {})
    out = {
        "status": "ok",
        "model_trained": True,
        "champion": m.metrics.get("champion"),
        "data_source": m.data_source,
        "threshold": m.threshold,
        "calibrated": m.calibrator is not None,
        "calibration_method": cal.get("method"),
        "band_edges": list(m.band_edges),
        "cv_f1_mean": m.metrics.get("cv", {}).get("f1_mean"),
        "cv_auc_mean": m.metrics.get("cv", {}).get("auc_mean"),
    }
    if m.data_source == "generated":
        out["data_warning"] = ("Model trained on a reproducible generated dataset "
                               "matching the competition schema, not a real portfolio.")
    return out


@app.get("/schema")
def schema():
    m = get_model()
    lo, hi = m.band_edges
    return {
        "raw_features": [
            "YearOfObservation", "Insured_Period (0.5 or 1.0)", "Residential (0/1)",
            "Building_Painted (N/V)", "Building_Fenced (N/V)", "Garden (V/O, nullable)",
            "Settlement (U/R)", "Building Dimension (m², nullable)", "Building_Type",
            "NumberOfWindows (without/1..9/>=10)", "Geo_Code (nullable)",
        ],
        "building_types": ["Fire-resistive", "Non-combustible", "Ordinary", "Wood-framed"],
        "risk_bands": {
            BAND_NAMES[0]: f"< {lo:.3f} (below the portfolio claim rate)",
            BAND_NAMES[1]: f"{lo:.3f} – {hi:.3f} (up to 2× the portfolio rate)",
            BAND_NAMES[2]: f">= {hi:.3f} (more than 2× the portfolio rate)",
        },
        "probability_scale": "calibrated" if m.calibrator is not None else "raw",
    }


@app.post("/predict", response_model=Prediction)
def predict(building: Building):
    return _score([building])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.buildings:
        raise HTTPException(status_code=400, detail="Empty building list.")
    results = _score(req.buildings)
    bands = pd.Series([r.risk_level for r in results]).value_counts().to_dict()
    return BatchResponse(
        count=len(results),
        results=results,
        summary={
            "predicted_claims": int(sum(r.prediction == "oui" for r in results)),
            "risk_bands": bands,
        },
    )

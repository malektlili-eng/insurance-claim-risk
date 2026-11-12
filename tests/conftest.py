"""Shared fixtures for the test suite. Run `python train.py` before `pytest`."""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.make_dataset import generate  # noqa: E402
from src_pipeline import read_raw_csv, TrainedModel  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")
HAS_MODEL = os.path.exists(os.path.join(MODEL_DIR, "model.joblib"))
needs_model = pytest.mark.skipif(not HAS_MODEL,
                                 reason="model not trained; run `python train.py` first")

SAMPLE = {
    "YearOfObservation": 2014, "Insured_Period": 0.5, "Residential": 1,
    "Building_Painted": "N", "Building_Fenced": "N", "Garden": "O",
    "Settlement": "R", "Building Dimension": 8000.0,
    "Building_Type": "Wood-framed", "NumberOfWindows": "without",
    "Geo_Code": "0042",
}


def raw_rows_as_json(df: pd.DataFrame) -> list:
    """Raw CSV rows -> API payloads (NaN -> null), exactly as a client would send."""
    cols = ["YearOfObservation", "Insured_Period", "Residential", "Building_Painted",
            "Building_Fenced", "Garden", "Settlement", "Building Dimension",
            "Building_Type", "NumberOfWindows", "Geo_Code"]
    out = []
    for rec in df[cols].to_dict(orient="records"):
        out.append({k: (None if (isinstance(v, float) and np.isnan(v)) else
                        (v.item() if hasattr(v, "item") else v))
                    for k, v in rec.items()})
    return out


@pytest.fixture(scope="session")
def data():
    tr, te = generate()
    return read_raw_csv(tr), read_raw_csv(te)


@pytest.fixture(scope="session")
def model():
    if not HAS_MODEL:
        pytest.skip("model not trained; run `python train.py` first")
    return TrainedModel.load(MODEL_DIR)


@pytest.fixture(scope="session")
def metadata():
    if not HAS_MODEL:
        pytest.skip("model not trained; run `python train.py` first")
    with open(os.path.join(MODEL_DIR, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def oof():
    path = os.path.join(MODEL_DIR, "oof_predictions.csv")
    if not os.path.exists(path):
        pytest.skip("oof_predictions.csv missing; run `python train.py` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def client():
    if not HAS_MODEL:
        pytest.skip("model not trained; run `python train.py` first")
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)

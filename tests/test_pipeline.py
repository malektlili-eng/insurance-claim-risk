"""
Core pipeline tests: data contract, preprocessing invariants, leakage controls,
model behaviour and the API contract.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, SAMPLE, needs_model
from src_pipeline import (fit_preprocess, transform, oof_geo_encoding,
                          fit_transform_train, MODEL_FEATURES, TRIAGE_ACTION,
                          GEO_SMOOTHING)

RAW_COLS = [
    "Customer Id", "YearOfObservation", "Insured_Period", "Residential",
    "Building_Painted", "Building_Fenced", "Garden", "Settlement",
    "Building Dimension", "Building_Type", "NumberOfWindows", "Geo_Code",
]


# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
def test_data_schema(data):
    train, test = data
    assert list(train.columns) == RAW_COLS + ["Claim"]
    assert list(test.columns) == RAW_COLS
    assert len(train) == 5012 and len(test) == 2147


def test_class_imbalance_present(data):
    train, _ = data
    assert 0.18 < (train["Claim"] == "oui").mean() < 0.27


def test_missingness_realistic(data):
    train, _ = data
    assert train["Building Dimension"].isna().sum() > 0
    assert train["Geo_Code"].isna().sum() > 0


def test_geo_codes_are_read_as_strings(data):
    train, _ = data
    non_null = train["Geo_Code"].dropna()
    assert non_null.map(type).eq(str).all()
    assert non_null.nunique() > 500


# --------------------------------------------------------------------------- #
# Preprocessing invariants
# --------------------------------------------------------------------------- #
def test_transform_produces_no_nan(data):
    train, test = data
    st = fit_preprocess(train)
    for df in (train, test):
        assert transform(df, st)[MODEL_FEATURES].isna().sum().sum() == 0


def test_transform_feature_order_is_fixed(data):
    train, _ = data
    out = transform(train, fit_preprocess(train))
    assert list(out[MODEL_FEATURES].columns) == MODEL_FEATURES
    assert "Customer Id" not in out.columns and "Geo_Code" not in out.columns


def test_unseen_geo_code_falls_back_to_global_rate(data):
    train, _ = data
    st = fit_preprocess(train)
    out = transform(pd.DataFrame([dict(SAMPLE, Geo_Code="ZZ_NEVER_SEEN")]), st)
    assert out["Geo_Code_enc"].iloc[0] == pytest.approx(st.global_rate)


def test_missing_values_at_inference_are_imputed(data):
    train, _ = data
    st = fit_preprocess(train)
    row = dict(SAMPLE, **{"Building Dimension": np.nan, "Garden": None, "Geo_Code": None})
    out = transform(pd.DataFrame([row]), st)
    assert out[MODEL_FEATURES].isna().sum().sum() == 0
    assert out["Building Dimension"].iloc[0] == pytest.approx(st.dim_median)
    assert out["Geo_Code_enc"].iloc[0] == pytest.approx(st.global_rate)


@pytest.mark.parametrize("level,expected", [("without", 0), (">=10", 10), ("5", 5)])
def test_window_encoding(data, level, expected):
    train, _ = data
    out = transform(pd.DataFrame([dict(SAMPLE, NumberOfWindows=level)]), fit_preprocess(train))
    assert out["NumberOfWindows"].iloc[0] == expected


def test_dimension_is_capped_at_p99(data):
    train, _ = data
    st = fit_preprocess(train)
    out = transform(pd.DataFrame([dict(SAMPLE, **{"Building Dimension": 1e6})]), st)
    assert out["Building Dimension"].iloc[0] <= st.dim_cap99


# --------------------------------------------------------------------------- #
# Leakage controls
# --------------------------------------------------------------------------- #
def _toy(n=400, seed=0, unique_geo=True):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "YearOfObservation": 2014, "Insured_Period": 1.0, "Residential": 1,
        "Building_Painted": "V", "Building_Fenced": "V", "Garden": "V",
        "Settlement": "U", "Building Dimension": 1000.0,
        "Building_Type": "Ordinary", "NumberOfWindows": "3",
        "Geo_Code": ([f"Z{i}" for i in range(n)] if unique_geo
                     else rng.choice(["A", "B", "C"], n)),
        "Claim": np.where(rng.random(n) < 0.25, "oui", "non"),
    })


def test_training_rows_never_see_their_own_label():
    """With one building per zone, an in-sample encoding would equal
    (y + prior*m)/(1+m) — i.e. it would reveal the label. Out-of-fold, every
    building's zone is unseen, so the encoding cannot depend on its own label."""
    df = _toy(unique_geo=True)
    enc = oof_geo_encoding(df)
    y = (df["Claim"] == "oui").to_numpy()
    assert abs(enc[y].mean() - enc[~y].mean()) < 0.02
    in_sample = fit_preprocess(df).geo_target_map
    leaked = np.array([in_sample[g] for g in df["Geo_Code"]])
    assert leaked[y].mean() - leaked[~y].mean() > 1 / (1 + GEO_SMOOTHING) - 0.01


def test_validation_zone_unseen_in_training_gets_prior(data):
    """A zone that only exists in validation rows must not leak into features."""
    train, _ = data
    tr, val = train.iloc[:4000].copy(), train.iloc[4000:].copy()
    val.iloc[0, val.columns.get_loc("Geo_Code")] = "ONLY_IN_VALIDATION"
    st = fit_preprocess(tr)
    assert "ONLY_IN_VALIDATION" not in st.geo_target_map
    enc = transform(val.iloc[[0]], st)["Geo_Code_enc"].iloc[0]
    assert enc == pytest.approx(st.global_rate)


def test_fit_transform_train_uses_out_of_fold_geo(data):
    train, _ = data
    sub = train.iloc[:1500]
    st, X, _ = fit_transform_train(sub)
    geo_col = MODEL_FEATURES.index("Geo_Code_enc")
    in_sample = transform(sub, st)["Geo_Code_enc"].to_numpy()
    assert not np.allclose(X[:, geo_col], in_sample)
    np.testing.assert_allclose(X[:, geo_col], oof_geo_encoding(sub))


def test_target_encoding_is_smoothed(data):
    train, _ = data
    vals = np.array(list(fit_preprocess(train).geo_target_map.values()))
    assert vals.min() > 0.0 and vals.max() < 1.0


@needs_model
def test_cv_auc_does_not_exceed_theoretical_ceiling():
    """On generated data no honest model can beat the generator's own formula.
    The old, leaky encoding did; the current pipeline must not."""
    path = os.path.join(ROOT, "models", "leakage_study.json")
    if not os.path.exists(path):
        pytest.skip("leakage_study.json missing; run `python train.py`")
    with open(path) as f:
        study = json.load(f)
    if study["auc_oracle_generating_formula"] is None:
        pytest.skip("ceiling only defined for generated data")
    ceiling = study["auc_oracle_generating_formula"]
    assert study["auc_fold_safe"] <= ceiling + 0.01
    assert study["auc_leaky_geo_encoding"] > ceiling   # documents the bug that was fixed


@needs_model
def test_reported_cv_auc_is_below_ceiling(metadata):
    path = os.path.join(ROOT, "models", "leakage_study.json")
    with open(path) as f:
        ceiling = json.load(f)["auc_oracle_generating_formula"]
    if ceiling is not None:
        assert metadata["metrics"]["cv"]["auc_mean"] <= ceiling + 0.01


# --------------------------------------------------------------------------- #
# Model behaviour
# --------------------------------------------------------------------------- #
@needs_model
def test_score_record_contract(model):
    res = model.score_record(SAMPLE)
    assert 0.0 <= res["claim_probability"] <= 1.0
    assert res["prediction"] in ("oui", "non")
    assert res["recommended_action"] == TRIAGE_ACTION[res["risk_level"]]
    assert res["prediction"] == ("oui" if res["claim_probability"] >= model.threshold
                                 else "non") or abs(res["claim_probability"]
                                                    - model.threshold) < 1e-4


@needs_model
def test_scores_are_deterministic(model):
    assert model.score_record(SAMPLE) == model.score_record(SAMPLE)


@needs_model
def test_risk_ordering_is_sensible(model):
    safe = dict(SAMPLE, Building_Type="Fire-resistive", Building_Fenced="V", Garden="V",
                Settlement="U", Building_Painted="V", Insured_Period=1.0, Residential=0,
                NumberOfWindows="4", **{"Building Dimension": 400.0})
    assert (model.score_record(SAMPLE)["claim_probability"]
            > model.score_record(safe)["claim_probability"])


@needs_model
def test_metrics_are_reported(metadata):
    m = metadata["metrics"]
    for key in ("champion", "cv", "calibration", "bands", "oof_at_threshold",
                "gains_top30", "data_source", "leaderboard"):
        assert key in m
    assert 0.0 < metadata["threshold"] < 1.0


# --------------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------------- #
@needs_model
def test_health(client):
    body = client.get("/health").json()
    assert body["model_trained"] is True
    assert body["calibrated"] is True
    assert len(body["band_edges"]) == 2


@needs_model
def test_schema_endpoint(client):
    body = client.get("/schema").json()
    assert set(body["risk_bands"]) == {"Faible", "Moyen", "Élevé"}
    assert body["probability_scale"] == "calibrated"


@needs_model
def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert {"claim_probability", "prediction", "risk_level", "recommended_action",
            "threshold_used"} <= set(r.json())


@needs_model
def test_predict_accepts_nulls(client):
    payload = dict(SAMPLE, **{"Building Dimension": None, "Garden": None, "Geo_Code": None})
    assert client.post("/predict", json=payload).status_code == 200


@needs_model
def test_predict_accepts_numeric_geo_code(client):
    a = client.post("/predict", json=dict(SAMPLE, Geo_Code=42)).json()
    b = client.post("/predict", json=dict(SAMPLE, Geo_Code="0042")).json()
    assert a == b


@needs_model
def test_predict_batch(client):
    r = client.post("/predict/batch", json={"buildings": [SAMPLE, SAMPLE]})
    assert r.status_code == 200 and r.json()["count"] == 2


@needs_model
def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"buildings": []}).status_code == 400


@needs_model
@pytest.mark.parametrize("field,value", [
    ("Building_Type", "Concrete"), ("Insured_Period", 5.0), ("NumberOfWindows", "12"),
    ("Building Dimension", -10), ("Settlement", "X"),
])
def test_invalid_input_rejected(client, field, value):
    assert client.post("/predict", json=dict(SAMPLE, **{field: value})).status_code == 422

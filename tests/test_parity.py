"""
Train/serve parity: every consumer of the model must return identical answers.

"Single source of truth" is only a claim until something checks it. These tests
compare, row by row on the full 2,147-building test portfolio:

    train.py (predictions_test.csv)  ==  live API /predict/batch
    notebook (notebook_predictions.csv)  ==  live API /predict/batch
    API /predict (single)  ==  API /predict/batch
    model saved to disk  ==  model held in memory after training

An earlier version failed this badly: Geo codes were float keys in memory and
string keys after the JSON round-trip, so the deployed API ignored location and
disagreed with the published predictions on 710 of 2,147 buildings.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, SAMPLE, needs_model, raw_rows_as_json
from src_pipeline import (fit_preprocess, transform, canon_geo, PreprocessState,
                          read_raw_csv)

PRED_TRAIN_PY = os.path.join(ROOT, "predictions_test.csv")
PRED_NOTEBOOK = os.path.join(ROOT, "notebooks", "notebook_predictions.csv")
NOTEBOOK = os.path.join(ROOT, "notebooks", "Insurance_Claim_Risk_Case_Study.ipynb")
COMPARE = ["claim_probability", "prediction", "risk_level", "recommended_action"]


def _api_scores(client, raw: pd.DataFrame) -> pd.DataFrame:
    r = client.post("/predict/batch", json={"buildings": raw_rows_as_json(raw)})
    assert r.status_code == 200, r.text
    return pd.DataFrame(r.json()["results"])


def _assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert len(a) == len(b)
    np.testing.assert_array_equal(a["claim_probability"].to_numpy(),
                                  b["claim_probability"].to_numpy())
    for col in COMPARE[1:]:
        mismatches = int((a[col].to_numpy() != b[col].to_numpy()).sum())
        assert mismatches == 0, f"{mismatches} rows disagree on {col}"


# --------------------------------------------------------------------------- #
# Cross-consumer parity on the full test portfolio
# --------------------------------------------------------------------------- #
@needs_model
def test_api_matches_train_py_on_every_test_building(client, data):
    _, test = data
    published = pd.read_csv(PRED_TRAIN_PY)
    assert list(published["customer_id"]) == list(test["Customer Id"])
    _assert_same(_api_scores(client, test), published)


@needs_model
def test_api_matches_notebook_on_every_test_building(client, data):
    if not os.path.exists(PRED_NOTEBOOK):
        pytest.skip("notebook not executed; see notebooks/build_notebook.py")
    _, test = data
    notebook = pd.read_csv(PRED_NOTEBOOK)
    assert list(notebook["customer_id"]) == list(test["Customer Id"])
    _assert_same(_api_scores(client, test), notebook)


@needs_model
def test_api_single_endpoint_matches_batch(client, data):
    _, test = data
    sample = test.sample(25, random_state=0)
    batch = _api_scores(client, sample)
    for i, payload in enumerate(raw_rows_as_json(sample)):
        single = client.post("/predict", json=payload)
        assert single.status_code == 200, single.text
        for col in COMPARE:
            assert single.json()[col] == batch.iloc[i][col]


@needs_model
def test_rows_with_missing_values_are_scored_by_api(client, data):
    """The API must score what the pipeline can score — including blanks."""
    _, test = data
    incomplete = test[test[["Building Dimension", "Garden", "Geo_Code"]].isna().any(axis=1)]
    assert len(incomplete) > 50
    published = pd.read_csv(PRED_TRAIN_PY).set_index("customer_id")
    expected = published.loc[incomplete["Customer Id"]].reset_index(drop=True)
    _assert_same(_api_scores(client, incomplete), expected)


@needs_model
def test_model_score_frame_matches_score_record(model):
    frame = model.score_frame(pd.DataFrame([SAMPLE])).iloc[0]
    record = model.score_record(SAMPLE)
    for col in COMPARE:
        assert record[col] == frame[col]


# --------------------------------------------------------------------------- #
# Regression tests for the Geo key bug that broke parity
# --------------------------------------------------------------------------- #
def test_geo_codes_are_canonical():
    s = pd.Series(["0042", "42", 42, 42.0, " 42 ", "2A065", None, np.nan, "", "nan"])
    out = canon_geo(s).tolist()
    assert out[:5] == ["42"] * 5
    assert out[5] == "2A065"
    assert all(pd.isna(v) for v in out[6:])


def test_preprocess_state_survives_json_round_trip(data, tmp_path):
    """Saving and reloading the state must not change a single feature value."""
    train, test = data
    st = fit_preprocess(train)
    path = tmp_path / "state.json"
    st.to_json(str(path))
    reloaded = PreprocessState.from_json(str(path))
    before = transform(test, st)
    after = transform(test, reloaded)
    pd.testing.assert_frame_equal(before, after)


def test_geo_encoding_is_used_after_reload(data, tmp_path):
    """Exactly the failure seen before: after reload every row fell back to the
    global rate. Only rows with a missing or never-seen code may fall back now."""
    train, test = data
    st = fit_preprocess(train)
    path = tmp_path / "state.json"
    st.to_json(str(path))
    enc = transform(test, PreprocessState.from_json(str(path)))["Geo_Code_enc"]
    codes = canon_geo(test["Geo_Code"])
    known = codes.isin(st.geo_target_map.keys())
    assert known.mean() > 0.5
    assert not np.allclose(enc[known.to_numpy()], st.global_rate)


def test_geo_code_format_does_not_change_score(model):
    scores = {fmt: model.score_record(dict(SAMPLE, Geo_Code=fmt))["claim_probability"]
              for fmt in ["0042", "42", 42]}
    assert len(set(scores.values())) == 1, scores


@needs_model
def test_saved_bundle_matches_freshly_trained_model(model, data):
    """predictions_test.csv is written from the in-memory model right after
    training; the API loads the saved bundle. They must agree exactly."""
    _, test = data
    reloaded = model.score_frame(test)
    published = pd.read_csv(PRED_TRAIN_PY)
    _assert_same(reloaded, published)


# --------------------------------------------------------------------------- #
# Structural guarantees: nobody re-implements the pipeline on the side
# --------------------------------------------------------------------------- #
def _code_cells(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_defines_no_pipeline_logic_of_its_own():
    src = _code_cells(NOTEBOOK)
    forbidden = ["def ", "predict_proba(", "StandardScaler", "SMOTE(",
                 "fit_preprocess(", "geo_target_map", ".fit("]
    found = [tok for tok in forbidden if tok in src]
    assert not found, f"notebook re-implements pipeline steps: {found}"
    assert "score_frame(" in src
    assert "from src_pipeline import" in src


@pytest.mark.parametrize("rel", ["api/main.py", "app/streamlit_app.py", "train.py"])
def test_consumers_score_only_through_score_frame(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        src = f.read()
    assert "score_frame(" in src or "score_record(" in src
    for tok in ["predict_proba(", "fit_preprocess(", "transform(test", "geo_target_map"]:
        assert tok not in src, f"{rel} bypasses score_frame via {tok!r}"


# --------------------------------------------------------------------------- #
# Optional end-to-end: re-execute the notebook, then compare with the API
# --------------------------------------------------------------------------- #
@needs_model
@pytest.mark.skipif(os.environ.get("RUN_NOTEBOOK") != "1",
                    reason="slow; set RUN_NOTEBOOK=1 to re-execute the notebook")
def test_freshly_executed_notebook_matches_api(client, data, tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                    "--execute", "--output", str(out), NOTEBOOK,
                    "--ExecutePreprocessor.timeout=1200"],
                   check=True, cwd=os.path.dirname(NOTEBOOK), capture_output=True)
    _, test = data
    _assert_same(_api_scores(client, test), pd.read_csv(PRED_NOTEBOOK))

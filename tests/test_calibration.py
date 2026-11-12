"""
Probability calibration: the deployed scores must mean what they say.

Checks the claims made in the README against the saved artefacts
(models/metadata.json, models/oof_predictions.csv) and unit-tests the
calibrators themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src_pipeline import (SigmoidCalibrator, fit_calibrator, crossfit_calibrate,
                          expected_calibration_error, best_threshold, risk_band,
                          BAND_NAMES, BAND_MULTIPLES)


# --------------------------------------------------------------------------- #
# Claims about the deployed model
# --------------------------------------------------------------------------- #
def test_bundle_is_calibrated(model, metadata):
    assert model.calibrator is not None
    assert metadata["calibrated"] is True


def test_mean_calibrated_probability_equals_claim_rate(oof):
    assert abs(oof["p_calibrated"].mean() - oof["y"].mean()) < 0.01


def test_raw_smote_scores_overstate_risk(oof):
    """The reason calibration exists: SMOTE-trained scores sit far above the base rate."""
    assert oof["p_raw"].mean() > 1.5 * oof["y"].mean()


def test_calibration_improves_every_metric(metadata):
    cal = metadata["metrics"]["calibration"]
    assert cal["ece_calibrated"] < cal["ece_raw"] / 5
    assert cal["brier_calibrated"] < cal["brier_raw"]
    deployed = cal["comparison"][cal["method"]]
    assert deployed["log_loss"] < cal["log_loss_raw"]


def test_deployed_method_has_lowest_held_out_log_loss(metadata):
    cal = metadata["metrics"]["calibration"]
    best = min(cal["comparison"], key=lambda k: cal["comparison"][k]["log_loss"])
    assert cal["method"] == best


def test_calibrated_scores_are_valid_probabilities(model, data):
    _, test = data
    p = model.predict_proba_df(test)
    assert np.all((p >= 0) & (p <= 1))


def test_calibrator_is_monotone(model):
    """Calibration must never reorder buildings by risk."""
    grid = np.linspace(0.001, 0.999, 500)
    out = model.calibrator.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)


def test_ranking_is_preserved_by_calibration(model, data):
    _, test = data
    raw = model.predict_raw_df(test)
    cal = model.predict_proba_df(test)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(cal[order]) >= -1e-12)


# --------------------------------------------------------------------------- #
# Risk bands on the calibrated scale
# --------------------------------------------------------------------------- #
def test_band_edges_are_multiples_of_claim_rate(model, metadata):
    base = metadata["metrics"]["base_rate"]
    assert model.band_edges == pytest.approx((base * BAND_MULTIPLES[0],
                                              base * BAND_MULTIPLES[1]))


def test_observed_claim_rate_rises_across_bands(metadata):
    bands = {b["band"]: b for b in metadata["metrics"]["bands"]}
    rates = [bands[name]["observed_rate"] for name in BAND_NAMES]
    assert rates[0] < rates[1] < rates[2], rates
    assert rates[0] < metadata["metrics"]["base_rate"] < rates[2]


def test_lower_bands_are_well_calibrated(metadata):
    """Faible and Moyen hold ~97% of buildings: predicted ≈ observed there."""
    for b in metadata["metrics"]["bands"]:
        if b["band"] in ("Faible", "Moyen"):
            assert abs(b["mean_predicted"] - b["observed_rate"]) < 0.03, b


def test_every_band_is_populated_on_test_portfolio(model, data):
    _, test = data
    counts = model.score_frame(test)["risk_level"].value_counts()
    for name in BAND_NAMES:
        assert counts.get(name, 0) > 0, f"band {name} empty on test portfolio"


def test_risk_band_uses_given_edges():
    edges = (0.2, 0.4)
    assert risk_band(0.19, edges) == "Faible"
    assert risk_band(0.20, edges) == "Moyen"
    assert risk_band(0.39, edges) == "Moyen"
    assert risk_band(0.40, edges) == "Élevé"


# --------------------------------------------------------------------------- #
# Threshold
# --------------------------------------------------------------------------- #
def test_threshold_meets_recall_guarantee_out_of_fold(model, oof):
    y = oof["y"].to_numpy()
    flagged = oof["p_calibrated"].to_numpy() >= model.threshold
    assert y[flagged].sum() / y.sum() >= 0.70


def test_best_threshold_respects_recall_constraint():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, 3000)
    p = np.clip(y * 0.3 + rng.normal(0.3, 0.15, 3000), 0.001, 0.999)
    for target in (0.6, 0.7, 0.9):
        thr = best_threshold(y, p, min_recall=target)
        assert ((p >= thr) & (y == 1)).sum() / y.sum() >= target - 1e-9


# --------------------------------------------------------------------------- #
# Calibrator unit tests
# --------------------------------------------------------------------------- #
def _shifted_scores(n=20000, shift=1.2, seed=0):
    """True probabilities + a SMOTE-like upward shift in logit space."""
    rng = np.random.default_rng(seed)
    logit_true = rng.normal(-1.3, 0.8, n)
    p_true = 1 / (1 + np.exp(-logit_true))
    y = rng.binomial(1, p_true)
    raw = 1 / (1 + np.exp(-(logit_true + shift)))
    return raw, y


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_removes_prior_shift(method):
    raw, y = _shifted_scores()
    assert expected_calibration_error(y, raw) > 0.15
    cal = fit_calibrator(raw[:10000], y[:10000], method).predict(raw[10000:])
    assert expected_calibration_error(y[10000:], cal) < 0.02


def test_sigmoid_calibrator_is_smooth_in_the_tail():
    raw, y = _shifted_scores()
    out = SigmoidCalibrator().fit(raw, y).predict(np.linspace(0.9, 0.999, 50))
    assert np.all(np.diff(out) > 0)          # strictly increasing: no flat steps


def test_crossfit_calibration_never_sees_its_own_fold_labels():
    """Scores of a fold must not change when that fold's labels are shuffled."""
    raw, y = _shifted_scores(n=4000)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(raw, y))
    base = crossfit_calibrate(raw, y, folds, "sigmoid")
    held = folds[0][1]
    y_perm = y.copy()
    y_perm[held] = np.random.default_rng(1).permutation(y[held])
    again = crossfit_calibrate(raw, y_perm, folds, "sigmoid")
    np.testing.assert_allclose(base[held], again[held])


def test_ece_is_zero_for_perfect_calibration():
    levels = np.arange(5, 100, 10) / 100          # 0.05, 0.15, ..., 0.95 (one per bin)
    p = np.repeat(levels, 1000)
    y = np.concatenate([np.r_[np.ones(round(v * 1000)), np.zeros(1000 - round(v * 1000))]
                        for v in levels])
    assert expected_calibration_error(y, p) < 1e-9

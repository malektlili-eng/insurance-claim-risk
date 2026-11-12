"""
Leakage study: how much of a reported AUC is signal, and how much is leakage?
================================================================================

Compares three ways of building the Geo_Code target encoding under the same
5-fold cross-validation (SMOTE + scaling refit per fold, same champion model):

  leaky        target map fitted on ALL rows before splitting — each validation
               row's encoding contains its own label (what an earlier version of
               this project did)
  fold_safe    map fitted on training folds only; training rows get an
               out-of-fold encoding (what the pipeline does now)
  no_geo       fold-safe, with the Geo features removed entirely

For generated data it also scores the generator's own noise-free formula:
no model can beat that ceiling, so any AUC above it must be leakage.

Run:  python diagnostics.py      (writes models/leakage_study.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data.make_dataset import generate, dataset_provenance, true_risk_logit  # noqa: E402
from src_pipeline import (read_raw_csv, fit_preprocess, transform,           # noqa: E402
                          fit_transform_train, oof_geo_encoding, _fit_balanced,
                          _labels, MODEL_FEATURES, SEED)

GEO_FEATURES = ("Geo_Code_enc", "risk_score")


def _cv_auc(train, y, mode: str, seed: int) -> float:
    model = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    feats = [f for f in MODEL_FEATURES if not (mode == "no_geo" and f in GEO_FEATURES)]
    leaky_state = fit_preprocess(train) if mode == "leaky" else None
    oof = np.empty(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for a, b in skf.split(np.zeros(len(y)), y):
        if mode == "leaky":
            Xa = transform(train.iloc[a], leaky_state)[feats].to_numpy()
            Xb = transform(train.iloc[b], leaky_state)[feats].to_numpy()
        else:
            st, _, _ = fit_transform_train(train.iloc[a], seed=seed)
            Xa = transform(train.iloc[a], st,
                           geo_enc=oof_geo_encoding(train.iloc[a], seed=seed))[feats].to_numpy()
            Xb = transform(train.iloc[b], st)[feats].to_numpy()
        m, sc = _fit_balanced(model, Xa, y[a], seed)
        oof[b] = m.predict_proba(sc.transform(Xb))[:, 1]
    return float(roc_auc_score(y, oof))


def leakage_study(train_csv: str, seed: int = SEED) -> dict:
    train = read_raw_csv(train_csv)
    y = _labels(train)
    source = dataset_provenance(train_csv)
    out = {
        "data_source": source,
        "auc_leaky_geo_encoding": _cv_auc(train, y, "leaky", seed),
        "auc_fold_safe": _cv_auc(train, y, "fold_safe", seed),
        "auc_fold_safe_without_geo": _cv_auc(train, y, "no_geo", seed),
    }
    out["auc_oracle_generating_formula"] = (
        float(roc_auc_score(y, true_risk_logit(train))) if source == "generated" else None)
    return out


if __name__ == "__main__":
    tr, _ = generate()
    res = leakage_study(tr)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    with open(os.path.join(HERE, "models", "leakage_study.json"), "w") as f:
        json.dump(res, f, indent=2)
    for k, v in res.items():
        print(f"{k:32s} {v if isinstance(v, str) or v is None else round(v, 4)}")

"""
Core modelling pipeline for the Building Insurance Claim-Risk case study.
================================================================================

This module is the single source of truth for:
  * feature engineering / preprocessing,
  * the leakage-safe SMOTE + scaling + model pipeline,
  * probability calibration (Platt vs isotonic, chosen on held-out log-loss),
  * training, evaluation, persistence and inference.

Every consumer — `train.py`, the FastAPI service, the Streamlit app and the
notebook — scores buildings through ONE method, `TrainedModel.score_frame`.
`tests/test_parity.py` verifies that they all return identical outputs.

Design choices that matter for a claims-triage use case
-------------------------------------------------------
* We optimise for **recall on the claim class** at a tunable decision
  threshold: missing a genuine claim costs more than a false alarm.
* SMOTE trains the model on a 50/50 class mix, so its raw scores overstate risk
  on a portfolio where only ~22% of buildings claim. We therefore **calibrate**
  the scores on out-of-fold predictions — Platt scaling and isotonic regression
  are both cross-fitted and the one with lower held-out log-loss is deployed —
  so a score of 0.30 means "about 30% of such buildings claim". Risk bands are
  defined on that calibrated scale, as multiples of the portfolio claim rate.

Leakage controls
----------------
* All preprocessing statistics (imputation values, P99 cap, Geo target map) are
  learned from the training portion of each split only.
* Training rows receive an **out-of-fold** Geo target encoding, so no row's
  encoding ever contains its own label.
* SMOTE and scaling are refit inside every cross-validation fold.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score, accuracy_score,
                             precision_recall_curve, brier_score_loss, log_loss)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
RAW_FEATURES = [
    "YearOfObservation", "Insured_Period", "Residential", "Building_Painted",
    "Building_Fenced", "Garden", "Settlement", "Building Dimension",
    "Building_Type", "NumberOfWindows", "Geo_Code",
]
# Order of the engineered feature vector fed to the model.
MODEL_FEATURES = [
    "YearOfObservation", "Insured_Period", "Residential", "Building_Painted",
    "Building_Fenced", "Garden", "Settlement", "Building Dimension",
    "Building_Type", "NumberOfWindows", "Geo_Code_enc", "dim_per_window",
    "risk_score", "is_large", "short_period",
]
GLOBAL_CLAIM_RATE = 0.2247  # fallback before any data has been seen
GEO_SMOOTHING = 20.0        # pseudo-count pulling rare zones to the global rate
SEED = 42

# Read every raw CSV with this so Geo codes are never parsed as floats.
CSV_DTYPES = {"Geo_Code": str, "NumberOfWindows": str}

# Risk bands live on the CALIBRATED probability scale and are expressed as
# multiples of the portfolio's own claim rate: below average, up to twice the
# average, more than twice the average. Edges are computed at training time
# and stored in the model bundle.
BAND_MULTIPLES = (1.0, 2.0)
DEFAULT_BAND_EDGES = (GLOBAL_CLAIM_RATE * BAND_MULTIPLES[0],
                      GLOBAL_CLAIM_RATE * BAND_MULTIPLES[1])
BAND_NAMES = ("Faible", "Moyen", "Élevé")
TRIAGE_ACTION = {
    "Faible": "Auto-approve / fast-track underwriting",
    "Moyen": "Route to standard manual review",
    "Élevé": "Escalate to senior underwriter + request inspection",
}


def read_raw_csv(path: str) -> pd.DataFrame:
    """Load a raw dataset CSV with the dtypes the pipeline expects."""
    return pd.read_csv(path, dtype=CSV_DTYPES)


# --------------------------------------------------------------------------- #
# Geo code canonicalisation
# --------------------------------------------------------------------------- #
def _canon_geo_value(v) -> object:
    """'0042', '42', 42 and 42.0 all map to '42'; 'nan'/None/'' map to NaN."""
    if v is None:
        return np.nan
    if isinstance(v, float) and np.isnan(v):
        return np.nan
    t = str(v).strip()
    if t == "" or t.lower() in ("nan", "none", "null"):
        return np.nan
    try:
        fv = float(t)
        if np.isfinite(fv) and fv.is_integer():
            return str(int(fv))
    except ValueError:
        pass
    return t


def canon_geo(series: pd.Series) -> pd.Series:
    return pd.Series(series, copy=True).astype(object).map(_canon_geo_value)


def _smoothed_geo_map(geo: pd.Series, y: np.ndarray, prior: float) -> dict:
    """Claim rate per zone, shrunk toward `prior` (rare zones can't hit 0 or 1)."""
    frame = pd.DataFrame({"g": geo.values, "y": np.asarray(y, dtype=float)})
    frame = frame.dropna(subset=["g"])
    agg = frame.groupby("g")["y"].agg(["sum", "count"])
    smoothed = (agg["sum"] + prior * GEO_SMOOTHING) / (agg["count"] + GEO_SMOOTHING)
    return {str(k): float(v) for k, v in smoothed.items()}


# --------------------------------------------------------------------------- #
# Preprocessing artefacts
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessState:
    """Everything learned from a training split that is needed to transform new rows."""
    geo_target_map: dict = field(default_factory=dict)
    dim_median: float = 0.0
    dim_cap99: float = 0.0
    dim_q75: float = 0.0
    garden_mode: str = "V"
    global_rate: float = GLOBAL_CLAIM_RATE

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_json(cls, path: str) -> "PreprocessState":
        with open(path) as f:
            return cls(**json.load(f))


BINARY_MAPS = {
    "Building_Painted": {"N": 1, "V": 0},
    "Building_Fenced": {"N": 1, "V": 0},
    "Garden": {"V": 1, "O": 0},
    "Settlement": {"U": 1, "R": 0},
}
BUILDING_TYPE_MAP = {"Wood-framed": 3, "Ordinary": 2,
                     "Non-combustible": 1, "Fire-resistive": 0}


def _clean_windows(series: pd.Series) -> pd.Series:
    s = series.astype(object).replace({"without": 0, ">=10": 10})
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _labels(df: pd.DataFrame) -> np.ndarray:
    return (df["Claim"] == "oui").astype(int).to_numpy()


def fit_preprocess(train: pd.DataFrame) -> PreprocessState:
    """Learn imputation values, the Geo target map and derived thresholds.

    Call this on TRAINING rows only.
    """
    dim = pd.to_numeric(train["Building Dimension"], errors="coerce")
    cap = float(dim.quantile(0.99))
    st = PreprocessState(
        dim_median=float(dim.median()),
        dim_cap99=cap,
        dim_q75=float(dim.fillna(dim.median()).clip(upper=cap).quantile(0.75)),
        garden_mode=str(train["Garden"].mode()[0]),
        global_rate=float(_labels(train).mean()),
    )
    st.geo_target_map = _smoothed_geo_map(canon_geo(train["Geo_Code"]),
                                          _labels(train), st.global_rate)
    return st


def oof_geo_encoding(train: pd.DataFrame, n_splits: int = 5,
                     seed: int = SEED) -> np.ndarray:
    """Out-of-fold Geo target encoding for the training rows themselves.

    Each row is encoded with a map built from the OTHER folds, so a building's
    own label never leaks into its feature.
    """
    geo = canon_geo(train["Geo_Code"]).reset_index(drop=True)
    y = _labels(train)
    out = np.empty(len(train), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in skf.split(np.zeros(len(y)), y):
        prior = float(y[fit_idx].mean())
        mp = _smoothed_geo_map(geo.iloc[fit_idx], y[fit_idx], prior)
        out[enc_idx] = geo.iloc[enc_idx].map(mp).fillna(prior).to_numpy(dtype=float)
    return out


def transform(df_in: pd.DataFrame, st: PreprocessState,
              geo_enc: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Apply the learned preprocessing + feature engineering to any frame.

    `geo_enc` overrides the Geo encoding (used for out-of-fold training rows);
    when omitted the learned map is applied, which is what serving does.
    """
    df = df_in.copy().reset_index(drop=True)
    df = df.drop(columns=["Customer Id"], errors="ignore")

    df["NumberOfWindows"] = _clean_windows(df["NumberOfWindows"])

    df["Building Dimension"] = pd.to_numeric(df["Building Dimension"],
                                             errors="coerce").fillna(st.dim_median)
    df["Building Dimension"] = df["Building Dimension"].clip(upper=st.dim_cap99)

    df["Garden"] = df["Garden"].fillna(st.garden_mode)

    if geo_enc is not None:
        df["Geo_Code_enc"] = np.asarray(geo_enc, dtype=float)
    else:
        df["Geo_Code_enc"] = (canon_geo(df["Geo_Code"]).map(st.geo_target_map)
                              .astype(float).fillna(st.global_rate))
    df = df.drop(columns=["Geo_Code"], errors="ignore")

    for col, mp in BINARY_MAPS.items():
        df[col] = df[col].map(mp)
    df["Building_Type"] = df["Building_Type"].map(BUILDING_TYPE_MAP)

    # Engineered features
    df["dim_per_window"] = df["Building Dimension"] / (df["NumberOfWindows"] + 1)
    df["risk_score"] = (df["Building_Type"] * 0.4 + df["Residential"] * 0.2 +
                        (1 - df["Building_Fenced"]) * 0.2 + (1 - df["Garden"]) * 0.1 +
                        df["Geo_Code_enc"] * 0.1)
    df["is_large"] = (df["Building Dimension"] > st.dim_q75).astype(int)
    df["short_period"] = (df["Insured_Period"] == 0.5).astype(int)

    if "Claim" in df.columns:
        df["Claim"] = (df["Claim"] == "oui").astype(int)

    for c in MODEL_FEATURES:
        if c not in df.columns:
            df[c] = 0
    keep = MODEL_FEATURES + (["Claim"] if "Claim" in df.columns else [])
    return df[keep].fillna(0)


def fit_transform_train(train: pd.DataFrame, seed: int = SEED):
    """Fit preprocessing on `train` and return (state, X, y) for model fitting.

    Training rows get an out-of-fold Geo encoding; the returned state holds the
    full-training-set map used for every row scored later.
    """
    st = fit_preprocess(train)
    feats = transform(train, st, geo_enc=oof_geo_encoding(train, seed=seed))
    return st, feats[MODEL_FEATURES].to_numpy(), feats["Claim"].to_numpy()


# --------------------------------------------------------------------------- #
# Risk bands, threshold, calibration metrics
# --------------------------------------------------------------------------- #
def risk_band(p: float, edges=DEFAULT_BAND_EDGES) -> str:
    """Map a calibrated probability to Faible / Moyen / Élevé."""
    if p < edges[0]:
        return "Faible"
    if p < edges[1]:
        return "Moyen"
    return "Élevé"


def band_report(y: np.ndarray, p: np.ndarray, edges) -> list:
    """Size, mean predicted and OBSERVED claim rate per band (out-of-fold)."""
    bands = np.array([risk_band(float(v), edges) for v in p])
    rows = []
    for name in BAND_NAMES:
        mask = bands == name
        rows.append({
            "band": name,
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "mean_predicted": float(p[mask].mean()) if mask.any() else None,
            "observed_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   min_recall: float = 0.70) -> float:
    """
    Threshold that maximises F1 subject to recall on the claim class
    >= `min_recall` (we refuse to miss more than 1 - min_recall of real claims).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    ok = rec >= min_recall
    idx = int(np.argmax(np.where(ok, f1, -1))) if ok.any() else int(np.argmax(f1))
    return float(thr[idx])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Weighted mean |observed rate - mean predicted| over equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


class SigmoidCalibrator:
    """Platt scaling: a 1-D logistic regression on the logit of the raw score.

    For a model trained on SMOTE-balanced data the distortion is mostly a prior
    shift in logit space, which this corrects smoothly (no step artefacts in the
    sparse high-risk tail, unlike isotonic regression).
    """

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        self._lr = LogisticRegression(C=1e6, max_iter=1000)
        self._lr.fit(self._logit(raw).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(self._logit(raw).reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(raw: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


CALIBRATION_METHODS = ("sigmoid", "isotonic")


def fit_calibrator(raw: np.ndarray, y: np.ndarray, method: str = "sigmoid"):
    raw = np.asarray(raw, dtype=float)
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            raw, np.asarray(y, dtype=float))
    if method == "sigmoid":
        return SigmoidCalibrator().fit(raw, y)
    raise ValueError(f"unknown calibration method: {method}")


def crossfit_calibrate(raw: np.ndarray, y: np.ndarray, folds,
                       method: str = "sigmoid") -> np.ndarray:
    """Honest calibrated scores: each fold is calibrated by a map fitted on the others."""
    out = np.empty(len(raw), dtype=float)
    for fit_idx, app_idx in folds:
        out[app_idx] = fit_calibrator(raw[fit_idx], y[fit_idx], method).predict(raw[app_idx])
    return out


def _operating_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
    }


# --------------------------------------------------------------------------- #
# Trained-artefact bundle
# --------------------------------------------------------------------------- #
@dataclass
class TrainedModel:
    model: object
    scaler: StandardScaler
    state: PreprocessState
    threshold: float
    metrics: dict
    calibrator: Optional[object] = None   # SigmoidCalibrator or IsotonicRegression
    band_edges: tuple = DEFAULT_BAND_EDGES

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        joblib.dump(self.model, os.path.join(folder, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(folder, "scaler.joblib"))
        if self.calibrator is not None:
            joblib.dump(self.calibrator, os.path.join(folder, "calibrator.joblib"))
        self.state.to_json(os.path.join(folder, "preprocess_state.json"))
        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump({"threshold": self.threshold,
                       "band_edges": [float(e) for e in self.band_edges],
                       "calibrated": self.calibrator is not None,
                       "data_source": self.data_source,
                       "model_features": MODEL_FEATURES,
                       "metrics": self.metrics}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, folder: str) -> "TrainedModel":
        model = joblib.load(os.path.join(folder, "model.joblib"))
        scaler = joblib.load(os.path.join(folder, "scaler.joblib"))
        cal_path = os.path.join(folder, "calibrator.joblib")
        calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
        state = PreprocessState.from_json(os.path.join(folder, "preprocess_state.json"))
        with open(os.path.join(folder, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        edges = tuple(meta.get("band_edges", DEFAULT_BAND_EDGES))
        return cls(model, scaler, state, meta["threshold"], meta["metrics"],
                   calibrator, edges)

    @property
    def data_source(self) -> str:
        return self.metrics.get("data_source", "unknown")

    # ---- inference: every consumer goes through score_frame -------------- #
    def predict_raw_df(self, raw: pd.DataFrame) -> np.ndarray:
        """Uncalibrated model scores (trained on SMOTE-balanced data)."""
        feats = transform(raw, self.state)
        X = self.scaler.transform(feats[MODEL_FEATURES].to_numpy())
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_df(self, raw: pd.DataFrame) -> np.ndarray:
        """Calibrated claim probabilities."""
        p = self.predict_raw_df(raw)
        return self.calibrator.predict(p) if self.calibrator is not None else p

    def score_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """THE scoring path: probability, decision, risk band, action per row."""
        p = self.predict_proba_df(raw)
        bands = [risk_band(float(v), self.band_edges) for v in p]
        return pd.DataFrame({
            "claim_probability": np.round(p, 4),
            "prediction": np.where(p >= self.threshold, "oui", "non"),
            "risk_level": bands,
            "recommended_action": [TRIAGE_ACTION[b] for b in bands],
        })

    def score_record(self, record: dict) -> dict:
        row = self.score_frame(pd.DataFrame([record])).iloc[0]
        return {
            "claim_probability": float(row["claim_probability"]),
            "prediction": str(row["prediction"]),
            "risk_level": str(row["risk_level"]),
            "recommended_action": str(row["recommended_action"]),
            "threshold_used": self.threshold,
        }


# --------------------------------------------------------------------------- #
# Full training routine
# --------------------------------------------------------------------------- #
def _candidates(seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.1,
                                                   random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "K-NN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                        random_state=seed),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, random_state=seed,
                                     verbosity=0, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=seed,
                                       verbose=-1),
    }


def _leaderboard_row(name: str, y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "avg_precision": average_precision_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def _fit_balanced(model, X: np.ndarray, y: np.ndarray, seed: int):
    """SMOTE -> scale -> fit, all on the given training rows only."""
    X_sm, y_sm = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X, y)
    scaler = StandardScaler().fit(X_sm)
    fitted = clone(model).fit(scaler.transform(X_sm), y_sm)
    return fitted, scaler


def train_full(train_csv: str, seed: int = SEED, verbose: bool = True,
               data_source: str = "unknown") -> "TrainedModel":
    """Train, evaluate, calibrate and refit the deployable champion.

    The heavy evaluation arrays are attached as `tm.oof` (not persisted in the
    bundle; `train.py` writes them to models/oof_predictions.csv).
    """
    train = read_raw_csv(train_csv)
    y_all = _labels(train)

    # ---- 1. model selection on a held-out split ---------------------------
    idx = np.arange(len(train))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed,
                                       stratify=y_all)
    tr_raw, val_raw = train.iloc[tr_idx], train.iloc[val_idx]
    st_tr, X_tr, y_tr = fit_transform_train(tr_raw, seed=seed)
    val_feats = transform(val_raw, st_tr)
    X_val, y_val = val_feats[MODEL_FEATURES].to_numpy(), val_feats["Claim"].to_numpy()

    X_tr_sm, y_tr_sm = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X_tr, y_tr)
    scaler = StandardScaler().fit(X_tr_sm)
    X_tr_sc, X_val_sc = scaler.transform(X_tr_sm), scaler.transform(X_val)

    leaderboard, fitted = [], {}
    for name, mdl in _candidates(seed).items():
        mdl.fit(X_tr_sc, y_tr_sm)
        row = _leaderboard_row(name, y_val, mdl.predict_proba(X_val_sc)[:, 1])
        leaderboard.append(row)
        fitted[name] = mdl
        if verbose:
            print(f"  {name:22s} F1={row['f1']:.4f}  AUC={row['roc_auc']:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grids = [
        ("xgb", xgb.XGBClassifier(random_state=seed, verbosity=0, eval_metric="logloss"),
         {"n_estimators": [200, 300], "max_depth": [3, 4], "learning_rate": [0.05, 0.1]}),
        ("lgb", lgb.LGBMClassifier(random_state=seed, verbose=-1),
         {"n_estimators": [200, 300], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}),
        ("rf", RandomForestClassifier(random_state=seed, n_jobs=-1),
         {"n_estimators": [200], "max_depth": [None, 12], "max_features": ["sqrt", "log2"]}),
    ]
    tuned = [(k, GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=-1)
              .fit(X_tr_sc, y_tr_sm).best_estimator_) for k, est, grid in grids]
    voting = VotingClassifier(estimators=tuned, voting="soft", n_jobs=-1)
    voting.fit(X_tr_sc, y_tr_sm)
    row = _leaderboard_row("Voting Ensemble", y_val, voting.predict_proba(X_val_sc)[:, 1])
    leaderboard.append(row)
    fitted["Voting Ensemble"] = voting
    if verbose:
        print(f"  {'Voting Ensemble':22s} F1={row['f1']:.4f}  AUC={row['roc_auc']:.4f}")

    lb = (pd.DataFrame(leaderboard).sort_values("f1", ascending=False)
          .reset_index(drop=True))
    champ_name = lb.iloc[0]["model"]
    champ = fitted[champ_name]

    # ---- 2. out-of-fold scores for the champion on ALL training rows ------
    # Preprocessing, OOF Geo encoding, SMOTE and scaling are refit per fold.
    folds = list(cv.split(np.zeros(len(y_all)), y_all))
    oof_raw = np.empty(len(y_all), dtype=float)
    for f_tr, f_vl in folds:
        st_f, Xf, yf = fit_transform_train(train.iloc[f_tr], seed=seed)
        m_f, sc_f = _fit_balanced(champ, Xf, yf, seed)
        Xv = transform(train.iloc[f_vl], st_f)[MODEL_FEATURES].to_numpy()
        oof_raw[f_vl] = m_f.predict_proba(sc_f.transform(Xv))[:, 1]

    # ---- 3. calibration (cross-fitted, so the evaluation stays honest) ----
    # Both methods are evaluated on held-out folds; the lower log-loss wins.
    candidates_cal = {}
    for method in CALIBRATION_METHODS:
        p = crossfit_calibrate(oof_raw, y_all, folds, method)
        candidates_cal[method] = {
            "scores": p,
            "log_loss": float(log_loss(y_all, np.clip(p, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y_all, p)),
            "ece": expected_calibration_error(y_all, p),
        }
    cal_method = min(candidates_cal, key=lambda k: candidates_cal[k]["log_loss"])
    oof_cal = candidates_cal[cal_method]["scores"]
    thr = best_threshold(y_all, oof_cal, min_recall=0.70)

    fold_f1 = [float(f1_score(y_all[v], (oof_cal[v] >= thr).astype(int)))
               for _, v in folds]
    fold_auc = [float(roc_auc_score(y_all[v], oof_cal[v])) for _, v in folds]
    fold_auc_raw = [float(roc_auc_score(y_all[v], oof_raw[v])) for _, v in folds]

    base_rate = float(y_all.mean())
    band_edges = (base_rate * BAND_MULTIPLES[0], base_rate * BAND_MULTIPLES[1])
    order = np.argsort(-oof_cal, kind="stable")
    top30 = order[: int(round(0.30 * len(order)))]
    gains_at_30 = float(y_all[top30].sum() / max(1, y_all.sum()))

    metrics = {
        "champion": champ_name,
        "data_source": data_source,
        "n_train_rows": int(len(train)),
        "base_rate": base_rate,
        "leaderboard": lb.to_dict(orient="records"),
        "threshold": thr,
        "oof_at_threshold": _operating_metrics(y_all, oof_cal, thr),
        "flagged_share": float((oof_cal >= thr).mean()),
        "gains_top30": gains_at_30,
        "band_edges": [float(e) for e in band_edges],
        "bands": band_report(y_all, oof_cal, band_edges),
        "cv": {
            "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
            "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
            "auc_raw_mean": float(np.mean(fold_auc_raw)),
            "folds_f1": fold_f1,
        },
        "calibration": {
            "method": cal_method,
            "selection": "lowest cross-fitted log-loss on out-of-fold scores",
            "comparison": {k: {m: v[m] for m in ("log_loss", "brier", "ece")}
                           for k, v in candidates_cal.items()},
            "log_loss_raw": float(log_loss(y_all, np.clip(oof_raw, 1e-6, 1 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_all, oof_raw)),
            "brier_calibrated": float(brier_score_loss(y_all, oof_cal)),
            "ece_raw": expected_calibration_error(y_all, oof_raw),
            "ece_calibrated": expected_calibration_error(y_all, oof_cal),
            "mean_pred_raw": float(oof_raw.mean()),
            "mean_pred_calibrated": float(oof_cal.mean()),
            "base_rate": base_rate,
        },
    }

    # ---- 4. deployable artefacts: refit on all rows -----------------------
    calibrator = fit_calibrator(oof_raw, y_all, cal_method)
    st_all, X_all, y_fit = fit_transform_train(train, seed=seed)
    final_model, final_scaler = _fit_balanced(champ, X_all, y_fit, seed)

    tm = TrainedModel(final_model, final_scaler, st_all, thr, metrics, calibrator,
                      band_edges)
    tm.oof = pd.DataFrame({
        "y": y_all, "p_raw": oof_raw, "p_calibrated": oof_cal,
        **{f"p_{k}": v["scores"] for k, v in candidates_cal.items()},
    })
    return tm


if __name__ == "__main__":
    from data.make_dataset import generate, dataset_provenance  # type: ignore
    tr, _ = generate()
    tm = train_full(tr, data_source=dataset_provenance(tr))
    print("champion:", tm.metrics["champion"], "threshold:", round(tm.threshold, 3))
    print("CV F1:", round(tm.metrics["cv"]["f1_mean"], 4))

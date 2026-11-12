"""
Reproducible data layer for the Building Insurance Claim-Risk case study.
================================================================================

The underlying problem is the public **Data Science Nigeria / Zindi "Insurance
Prediction"** task: given 12 descriptors of an insured building, predict whether
it will file a claim during the insured period.

This module has ONE job: guarantee that `train_Insurance.csv` and
`test_Insurance.csv` exist in this folder so the whole pipeline (training,
evaluation, the FastAPI service and the Streamlit app) runs end-to-end on any
machine.

Resolution order
----------------
1. If the real Zindi CSVs are already present next to this file, they are used
   as-is and nothing is regenerated.
2. Otherwise a synthetic dataset is generated that matches the real schema
   *exactly* (column names, dtypes, categorical levels, class balance, missing-
   value rates, cardinality of Geo_Code, outliers in Building Dimension). The
   target is produced by a transparent logistic data-generating process so the
   learned relationships are meaningful rather than noise.

The generator is fully deterministic (fixed seed), so results are reproducible.

Provenance tracking
-------------------
Every reported metric must say which data it came from. When the generator
writes the CSVs it also writes `GENERATED_DATA.sha256` (a fingerprint of the
training CSV). `dataset_provenance()` returns "generated" only if the training
file on disk still matches that fingerprint; any other file — e.g. the real
competition CSV dropped in its place — is reported as "real". Training stores
the result in models/metadata.json, and the API, app and notebook display it.
"""
from __future__ import annotations

import hashlib
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "train_Insurance.csv")
TEST_PATH = os.path.join(HERE, "test_Insurance.csv")
FINGERPRINT_PATH = os.path.join(HERE, "GENERATED_DATA.sha256")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_provenance(train_csv: str = TRAIN_PATH) -> str:
    """'generated' if `train_csv` is the generator's own output, else 'real'."""
    fp = os.path.join(os.path.dirname(os.path.abspath(train_csv)),
                      os.path.basename(FINGERPRINT_PATH))
    if not os.path.exists(fp):
        return "real"
    with open(fp) as f:
        expected = f.read().split()[0].strip()
    return "generated" if _sha256(train_csv) == expected else "real"

# Real dataset dimensions (from the source task).
N_TRAIN = 5012
N_TEST = 2147
SEED = 42

# Categorical levels exactly as they appear in the source data.
BUILDING_TYPES = ["Fire-resistive", "Non-combustible", "Ordinary", "Wood-framed"]
WINDOW_LEVELS = ["without", "1", "2", "3", "4", "5", "6", "7", "8", "9", ">=10"]
PAINTED = ["N", "V"]        # N = not painted, V = painted
FENCED = ["N", "V"]
GARDEN = ["V", "O"]         # V = has garden, O = none
SETTLEMENT = ["U", "R"]     # Urban / Rural
YEARS = [2012, 2013, 2014, 2015, 2016]
PERIODS = [0.5, 1.0]


def _make_frame(n: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    """Build one raw frame (features + latent claim) with realistic structure."""
    customer_id = [f"H{start_id + i}" for i in range(n)]

    year = rng.choice(YEARS, size=n, p=[0.18, 0.20, 0.22, 0.20, 0.20])
    period = rng.choice(PERIODS, size=n, p=[0.35, 0.65])
    residential = rng.choice([0, 1], size=n, p=[0.55, 0.45])

    painted = rng.choice(PAINTED, size=n, p=[0.30, 0.70])
    fenced = rng.choice(FENCED, size=n, p=[0.45, 0.55])
    garden = rng.choice(GARDEN, size=n, p=[0.55, 0.45])
    settlement = rng.choice(SETTLEMENT, size=n, p=[0.62, 0.38])

    # Building Dimension: right-skewed, median ~1067, long tail with outliers.
    dim = rng.lognormal(mean=6.97, sigma=0.62, size=n)
    dim = np.clip(dim, 60, None)
    # inject a few extreme outliers (up to ~20,000) as in the real data
    n_out = max(1, int(0.01 * n))
    out_idx = rng.choice(n, size=n_out, replace=False)
    dim[out_idx] = rng.uniform(12000, 20840, size=n_out)

    btype = rng.choice(BUILDING_TYPES, size=n, p=[0.22, 0.28, 0.30, 0.20])

    # NumberOfWindows: ~49% "without", rest spread across 1..9 and >=10
    win_p = np.array([0.494, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03,
                      0.025, 0.02, 0.011])
    win_p = win_p / win_p.sum()
    windows = rng.choice(WINDOW_LEVELS, size=n, p=win_p)

    # Geo_Code: high-cardinality (~1115 unique codes)
    n_geo = 1115
    geo_pool = [f"{c:04d}" for c in range(n_geo)]
    geo = rng.choice(geo_pool, size=n)

    df = pd.DataFrame({
        "Customer Id": customer_id,
        "YearOfObservation": year.astype(int),
        "Insured_Period": period.astype(float),
        "Residential": residential.astype(int),
        "Building_Painted": painted,
        "Building_Fenced": fenced,
        "Garden": garden,
        "Settlement": settlement,
        "Building Dimension": dim.round(1),
        "Building_Type": btype,
        "NumberOfWindows": windows,
        "Geo_Code": geo,
    })

    # ---- Transparent data-generating process for the claim label ----------
    # Higher risk: wood-framed/ordinary, not fenced, no garden, rural,
    # residential, larger buildings, short insured period. Geo adds signal.
    type_risk = pd.Series(btype).map(
        {"Wood-framed": 1.15, "Ordinary": 0.55,
         "Non-combustible": -0.15, "Fire-resistive": -0.75}).to_numpy()
    geo_effect = (pd.Series(geo).astype(int).to_numpy() % 17) / 17.0 - 0.5
    dim_z = (np.log(dim) - np.log(dim).mean()) / np.log(dim).std()

    logit = (
        -1.55
        + type_risk
        + 0.45 * (fenced == "N")
        + 0.30 * (garden == "O")
        + 0.25 * (settlement == "R")
        + 0.28 * residential
        + 0.32 * dim_z
        + 0.40 * (period == 0.5)
        + 0.9 * geo_effect
        + rng.normal(0, 0.55, size=n)      # irreducible noise -> caps AUC realistically
    )
    prob = 1 / (1 + np.exp(-logit))
    claim = (rng.random(n) < prob).astype(int)
    df["_claim_latent"] = claim
    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Match the real missing-value footprint."""
    n = len(df)
    # Building Dimension ~1.54% missing
    idx = rng.choice(n, size=int(round(0.0154 * n)), replace=False)
    df.loc[idx, "Building Dimension"] = np.nan
    # Geo_Code ~1.46% missing
    idx = rng.choice(n, size=int(round(0.0146 * n)), replace=False)
    df.loc[idx, "Geo_Code"] = np.nan
    # Garden ~0.08% missing
    idx = rng.choice(n, size=max(1, int(round(0.0008 * n))), replace=False)
    df.loc[idx, "Garden"] = np.nan
    return df


def true_risk_logit(df: pd.DataFrame) -> np.ndarray:
    """The generator's noise-free risk formula, recomputed from observed columns.

    Only meaningful for GENERATED data. Its AUC is the ceiling any model can
    reach on this dataset (the labels also contain irreducible noise), which
    lets us tell real signal from leakage.
    """
    geo = pd.to_numeric(df["Geo_Code"], errors="coerce")
    geo_effect = np.where(geo.isna(), 0.0, (geo.fillna(0) % 17) / 17.0 - 0.5)
    dim = pd.to_numeric(df["Building Dimension"], errors="coerce")
    dim = dim.fillna(dim.median())
    dim_z = (np.log(dim) - np.log(dim).mean()) / np.log(dim).std()
    type_risk = df["Building_Type"].map(
        {"Wood-framed": 1.15, "Ordinary": 0.55,
         "Non-combustible": -0.15, "Fire-resistive": -0.75}).to_numpy()
    return (type_risk
            + 0.45 * (df["Building_Fenced"] == "N").to_numpy()
            + 0.30 * (df["Garden"] == "O").to_numpy()
            + 0.25 * (df["Settlement"] == "R").to_numpy()
            + 0.28 * df["Residential"].to_numpy()
            + 0.32 * dim_z.to_numpy()
            + 0.40 * (df["Insured_Period"] == 0.5).to_numpy()
            + 0.9 * geo_effect)


def generate(force: bool = False) -> tuple[str, str]:
    """Ensure train/test CSVs exist; return their paths."""
    if not force and os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        return TRAIN_PATH, TEST_PATH

    rng = np.random.default_rng(SEED)

    full = _make_frame(N_TRAIN + N_TEST, rng, start_id=1000)
    # Calibrate global positive rate to ~22.5% (real class balance).
    target_rate = 0.2247
    cur = full["_claim_latent"].mean()
    if cur > target_rate:
        pos = np.array(full.index[full["_claim_latent"] == 1])
        rng.shuffle(pos)
        n_flip = int((cur - target_rate) * len(full))
        full.loc[pos[:n_flip], "_claim_latent"] = 0

    train = full.iloc[:N_TRAIN].copy()
    test = full.iloc[N_TRAIN:].copy().reset_index(drop=True)

    # Train keeps the label as "oui"/"non"; test hides it (as in a competition).
    train["Claim"] = np.where(train["_claim_latent"] == 1, "oui", "non")
    train = train.drop(columns=["_claim_latent"])
    test = test.drop(columns=["_claim_latent"])

    train = _inject_missing(train, rng)
    test = _inject_missing(test, rng)

    # Reassign clean, contiguous ids per split.
    train["Customer Id"] = [f"H{1000 + i}" for i in range(len(train))]
    test["Customer Id"] = [f"H{20000 + i}" for i in range(len(test))]

    col_order = [
        "Customer Id", "YearOfObservation", "Insured_Period", "Residential",
        "Building_Painted", "Building_Fenced", "Garden", "Settlement",
        "Building Dimension", "Building_Type", "NumberOfWindows", "Geo_Code",
    ]
    train = train[col_order + ["Claim"]]
    test = test[col_order]

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    with open(FINGERPRINT_PATH, "w") as f:
        f.write(f"{_sha256(TRAIN_PATH)}  train_Insurance.csv\n")
    return TRAIN_PATH, TEST_PATH


if __name__ == "__main__":
    tr, te = generate(force=True)
    t = pd.read_csv(tr)
    print(f"train -> {tr}  shape={t.shape}")
    print(f"test  -> {te}")
    print("provenance:", dataset_provenance(tr))
    print("class balance:", t["Claim"].value_counts(normalize=True).round(4).to_dict())
    print("missing Building Dimension:", int(t["Building Dimension"].isna().sum()))
    print("missing Geo_Code:", int(t["Geo_Code"].isna().sum()))
    print("Geo_Code uniques:", t["Geo_Code"].nunique())

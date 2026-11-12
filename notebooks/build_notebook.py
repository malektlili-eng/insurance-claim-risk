"""
Builds notebooks/Insurance_Claim_Risk_Case_Study.ipynb.

Rules this notebook follows (enforced by tests/test_parity.py):
  * it defines no functions — all preprocessing, training, calibration and
    scoring come from src_pipeline / diagnostics;
  * it never calls predict_proba directly — buildings are scored only through
    TrainedModel.score_frame, the same method the API uses;
  * section 8 writes notebooks/notebook_predictions.csv, which the parity test
    compares row by row with the live API.

Rebuild + execute:
    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/Insurance_Claim_Risk_Case_Study.ipynb
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda t: cells.append(new_markdown_cell(t))      # noqa: E731
code = lambda t: cells.append(new_code_cell(t))        # noqa: E731

# --------------------------------------------------------------------------- 0
md("""# Building Insurance Claim-Risk Prediction — A Data-Science Case Study

**Problem.** From twelve descriptors of an insured building (structure, size,
location, coverage period), estimate the probability it files a **claim**
during the insured period, and turn that probability into an **underwriting
triage decision**.

**Why it is framed this way.** The two errors are not symmetric: a missed claim
means under-pricing and under-reserving; a false alarm costs an analyst a few
minutes. The work therefore optimises claim recall at a deliberately chosen
operating point, and calibrates the scores so a probability means what it says.

**Contents**
1. Data provenance & setup
2. Exploratory analysis
3. Features — and a leakage study that changed the headline numbers
4. Modelling
5. Evaluation (out-of-fold)
6. Probability calibration
7. Business interpretation: threshold, risk bands, triage policy
8. Deployment & train/serve parity
9. Conclusions and limitations

Every number is produced by `src_pipeline.py` / `diagnostics.py`, the code the
API and the Streamlit app run. This notebook defines no functions of its own.
""")

# --------------------------------------------------------------------------- 1
md("## 1. Data provenance & setup")
code("""import warnings, os, sys, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)
from data.make_dataset import generate, dataset_provenance
from src_pipeline import (read_raw_csv, train_full, TrainedModel,
                          MODEL_FEATURES, TRIAGE_ACTION, BAND_NAMES)
from diagnostics import leakage_study

train_csv, test_csv = generate()
SOURCE = dataset_provenance(train_csv)
train, test = read_raw_csv(train_csv), read_raw_csv(test_csv)

banner = {"generated": "⚠️  DATA SOURCE: GENERATED — a reproducible dataset matching the "
                       "competition schema. Every metric below describes this dataset, "
                       "not a real insurance portfolio.",
          "real": "DATA SOURCE: REAL competition dataset."}[SOURCE]
print(banner)
print(f"\\nTrain: {train.shape[0]:,} rows x {train.shape[1]} cols | "
      f"Test: {test.shape[0]:,} rows x {test.shape[1]} cols")
train.head()""")

md("""> **How provenance is decided.** `data/make_dataset.py` writes a SHA-256
> fingerprint whenever it generates the CSVs. If the training file on disk
> still matches it, the data is reported as *generated*; any other file — such
> as the real competition CSV dropped into `data/` — is reported as *real*. The
> flag is stored in the model bundle and shown by the API, the app and every
> figure.""")

code("""pd.DataFrame({
    "dtype":     train.dtypes.astype(str),
    "missing":   train.isnull().sum(),
    "missing_%": (train.isnull().mean() * 100).round(2),
    "unique":    train.nunique(),
})""")

# --------------------------------------------------------------------------- 2
md("""## 2. Exploratory analysis

- **Class imbalance:** about 78% no-claim vs 22% claim. Predicting "no claim"
  everywhere scores 78% accuracy and is useless for triage, so accuracy is not
  the target metric.
- **Building Dimension** is heavily right-skewed with outliers; median
  imputation plus a P99 cap keeps a few extremes from dominating the scaler.
- **Building_Type** carries the clearest signal.
- **Geo_Code** has ~1,100 levels for ~5,000 rows (≈4.5 buildings per zone).
  That ratio matters a great deal in section 3.""")

code("""y_flag = (train["Claim"] == "oui").astype(int)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
vc = train["Claim"].value_counts()
ax[0,0].bar(vc.index, vc.values, color=["#2c7fb8", "#c0392b"]); ax[0,0].set_title("Class balance")
for i, v in enumerate(vc.values):
    ax[0,0].text(i, v, f"{v/len(train):.1%}", ha="center", va="bottom")
ax[0,1].hist(pd.to_numeric(train["Building Dimension"], errors="coerce").dropna(), bins=60, color="#2c7fb8")
ax[0,1].set_yscale("log"); ax[0,1].set_title("Building Dimension (log count)")
r = train.assign(y=y_flag).groupby("Building_Type")["y"].mean().sort_values()
ax[0,2].barh(r.index, r.values, color="#c0392b"); ax[0,2].set_title("Claim rate by Building_Type")
r = train.assign(y=y_flag).groupby("Settlement")["y"].mean()
ax[1,0].bar(r.index, r.values, color="#2c7fb8"); ax[1,0].set_title("Claim rate by Settlement")
order = ["without"] + [str(i) for i in range(1, 10)] + [">=10"]
r = train.assign(y=y_flag).groupby("NumberOfWindows")["y"].mean().reindex(order)
ax[1,1].bar(range(len(r)), r.values, color="#c0392b"); ax[1,1].set_xticks(range(len(r)))
ax[1,1].set_xticklabels(r.index, rotation=45, fontsize=7); ax[1,1].set_title("Claim rate by NumberOfWindows")
miss = train.isna().mean(); miss = miss[miss > 0]
ax[1,2].barh(miss.index, miss.values * 100, color="#2c7fb8"); ax[1,2].set_title("Missing values (%)")
plt.tight_layout(); plt.show()
print("Buildings per Geo_Code (median):", int(train["Geo_Code"].value_counts().median()))""")

# --------------------------------------------------------------------------- 3
md("""## 3. Features — and a leakage study

Engineered features (defined once in `src_pipeline.transform`):

| Feature | Rationale |
|---|---|
| `Geo_Code_enc` | Smoothed claim rate of the building's zone (target encoding) |
| `risk_score` | Composite of type, fencing, garden, residential, zone |
| `dim_per_window` | Size per window, a construction-density proxy |
| `is_large` | Above the 75th-percentile footprint |
| `short_period` | Half-year coverage flag |

### The finding that changed the headline numbers

An earlier version of this project fitted the Geo target encoding on **all**
training rows *before* splitting, so each validation row's encoding contained
its own label. The cell below measures the effect under identical 5-fold
cross-validation. On generated data there is also a hard ceiling: the
generator's own noise-free formula. No honest model can beat it.""")

code("""study = leakage_study(train_csv)
rows = [("Geo map fitted on all rows (own label leaks)", study["auc_leaky_geo_encoding"]),
        ("Fold-safe + out-of-fold Geo encoding (current)", study["auc_fold_safe"]),
        ("Fold-safe, Geo features removed", study["auc_fold_safe_without_geo"])]
if study["auc_oracle_generating_formula"] is not None:
    rows.append(("CEILING: the generator's true formula, no noise", study["auc_oracle_generating_formula"]))
pd.DataFrame(rows, columns=["setup", "5-fold CV AUC"]).round(4)""")

md("""**Reading the table.** The leaky setup scores *above the theoretical
ceiling*, which is impossible without leakage. The whole difference was the
Geo encoding memorising labels. Fitted honestly, the Geo features add nothing
measurable: with ≈4.5 buildings per zone, smoothing (correctly) pulls almost
every zone back to the global rate. The fold-safe pipeline lands within ~0.02
of the ceiling, so the model is extracting nearly all the signal there is.

The fix, now in `src_pipeline`: the target map is fitted only on training
folds, and training rows receive an **out-of-fold** encoding, so no building's
feature ever contains its own label. Geo codes are also canonicalised (`'0042'`,
`42` and `42.0` are the same zone). A float/string mismatch had previously made
the deployed API ignore Geo entirely (section 8).""")

# --------------------------------------------------------------------------- 4
md("""## 4. Modelling

`train_full` runs the whole protocol:
1. stratified 80/20 split; preprocessing fitted on the 80% only; SMOTE and
   scaling on the training part only;
2. eight candidates compared on the held-out 20% (logistic regression → tuned
   gradient-boosted trees → a soft-voting ensemble);
3. the champion is re-evaluated **out-of-fold on every training row** (5 folds,
   with preprocessing, Geo encoding, SMOTE and scaling refit inside each fold);
4. its scores are calibrated (section 6), the threshold is chosen, and the
   deployable model is refit on all rows.""")

code("""tm = train_full(train_csv, verbose=True, data_source=SOURCE)
print("\\nChampion:", tm.metrics["champion"])""")

code("""pd.DataFrame(tm.metrics["leaderboard"]).round(4)""")

md("""**Why logistic regression wins.** With honest features the signal is weak and
close to additive. That's how the generator built it, and it's a common shape
in tabular insurance data too. Flexible ensembles have nothing extra to model
and spend their variance fitting SMOTE's synthetic minority points. The
regularised linear model generalises best and stays auditable: each
coefficient is a readable risk driver.""")

# --------------------------------------------------------------------------- 5
md("## 5. Evaluation (out-of-fold, all 5,012 training rows)")
code("""from sklearn.metrics import roc_curve, precision_recall_curve
oof = tm.oof
y, p = oof["y"].to_numpy(), oof["p_calibrated"].to_numpy()
cv = tm.metrics["cv"]
print(f"5-fold CV AUC: {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
print(f"5-fold CV F1 : {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}   (at the chosen threshold)")
print("Per-fold F1  :", [round(v, 4) for v in cv["folds_f1"]])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fpr, tpr, _ = roc_curve(y, p)
ax[0].plot(fpr, tpr, color="#c0392b", label=f"AUC {cv['auc_mean']:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].legend(); ax[0].set_title("ROC (out-of-fold)")
pr, rc, _ = precision_recall_curve(y, p)
ax[1].plot(rc, pr, color="#2c7fb8"); ax[1].axhline(y.mean(), ls="--", color="grey", label="base rate")
ax[1].legend(); ax[1].set_title("Precision–Recall (out-of-fold)")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
plt.tight_layout(); plt.show()""")

code("""coef = pd.Series(tm.model.coef_.ravel(), index=MODEL_FEATURES).sort_values()
coef.plot.barh(figsize=(7, 5), color=["#2c7fb8" if v < 0 else "#c0392b" for v in coef],
               title="Standardised coefficients (+ = more claim risk)")
plt.axvline(0, color="grey", lw=0.8); plt.tight_layout(); plt.show()""")

# --------------------------------------------------------------------------- 6
md("""## 6. Probability calibration

SMOTE trains the model on a 50/50 class mix, so its raw scores describe a world
where half of all buildings claim. On a portfolio where about 22% do, a raw
0.50 is not a 50% risk. Risk bands and any pricing use need probabilities that
match observed frequencies.

Two calibrators are fitted on the **out-of-fold** raw scores and evaluated
**cross-fitted** (each fold calibrated by a map learned on the other folds):
Platt scaling (a logistic curve on the raw logit, which suits the prior shift
SMOTE introduces) and isotonic regression (non-parametric). The one with lower
held-out log-loss, a proper scoring rule, is deployed.""")

code("""cal = tm.metrics["calibration"]
table = pd.DataFrame(
    [("raw (SMOTE-trained)", cal["log_loss_raw"], cal["brier_raw"], cal["ece_raw"], cal["mean_pred_raw"])] +
    [(f"calibrated · {k}", v["log_loss"], v["brier"], v["ece"], oof[f"p_{k}"].mean())
     for k, v in cal["comparison"].items()],
    columns=["scores", "log-loss", "Brier", "ECE", "mean predicted"]).round(4)
print(f"Observed claim rate: {cal['base_rate']:.4f}   |   deployed calibrator: {cal['method']}")
table""")

code("""bins = np.linspace(0, 1, 11)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for col, color, name in [("p_raw", "#c0392b", "raw"), ("p_sigmoid", "#2c7fb8", "Platt"),
                         ("p_isotonic", "#27ae60", "isotonic")]:
    s = oof[col].to_numpy(); idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
    pts = [(s[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 10]
    ax[0].plot(*zip(*pts), "o-", color=color, label=name)
ax[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
ax[0].set_xlabel("mean predicted (bin)"); ax[0].set_ylabel("observed claim rate (bin)")
ax[0].set_title("Reliability diagram (out-of-fold)"); ax[0].legend()
ax[1].hist(oof["p_raw"], bins=40, alpha=.55, color="#c0392b", label="raw")
ax[1].hist(oof["p_calibrated"], bins=40, alpha=.55, color="#2c7fb8", label="calibrated (deployed)")
ax[1].axvline(cal["base_rate"], color="black", ls="--", label="true base rate")
ax[1].legend(); ax[1].set_title("Score distribution")
plt.tight_layout(); plt.show()""")

md("""Raw scores overstate risk by roughly 2× across the whole range. After
calibration the mean predicted probability equals the observed claim rate,
and the reliability curve follows the diagonal. The remaining gap is at the
very top, where the few highest-scored buildings are somewhat over-predicted
(see the Élevé row in section 7). Platt and isotonic perform almost the same;
Platt wins narrowly and has no step artefacts in the sparse high-risk tail.""")

# --------------------------------------------------------------------------- 7
md("""## 7. Business interpretation

### 7.1 Threshold as a business lever
The decision threshold is chosen on calibrated out-of-fold scores to maximise
F1 **subject to recall ≥ 70%** on claims: the insurer refuses to let more than
~30% of real claims through without review. The cost of that guarantee, on a
signal this weak, is reviewing a large share of the portfolio. The trade-off is
stated explicitly here rather than hidden.""")

code("""op = tm.metrics["oof_at_threshold"]
print(f"Threshold (calibrated probability): {tm.threshold:.3f}")
print(f"Recall {op['recall']:.1%} | Precision {op['precision']:.1%} | F1 {op['f1']:.3f}")
print(f"Share of portfolio flagged for review: {tm.metrics['flagged_share']:.1%}")
for t in [0.15, tm.threshold, 0.25, 0.30, 0.40]:
    flag = p >= t
    print(f"  threshold {t:.3f}: review {flag.mean():5.1%} of buildings -> catch "
          f"{y[flag].sum() / y.sum():5.1%} of claims (precision {y[flag].mean():.1%})")""")

md("""### 7.2 Risk bands → triage policy
Bands are defined on the calibrated scale **relative to the portfolio's own
claim rate**, so they keep their meaning if the data changes:

| Band | Calibrated probability | Action |
|---|---|---|
| Faible | below the portfolio rate | Auto-approve / fast-track |
| Moyen | 1–2× the portfolio rate | Standard manual review |
| Élevé | above 2× the portfolio rate | Senior underwriter + inspection |

The table below compares each band's predicted probability with the
**observed** claim rate of the buildings in it (out-of-fold).""")

code("""bands = pd.DataFrame(tm.metrics["bands"])
bands["edges"] = [f"< {tm.band_edges[0]:.3f}", f"{tm.band_edges[0]:.3f}–{tm.band_edges[1]:.3f}",
                  f">= {tm.band_edges[1]:.3f}"]
bands["× portfolio rate (observed)"] = bands["observed_rate"] / tm.metrics["base_rate"]
bands["action"] = bands["band"].map(TRIAGE_ACTION)
bands[["band", "edges", "n", "share", "mean_predicted", "observed_rate",
       "× portfolio rate (observed)", "action"]].round(3)""")

md("""Observed risk rises monotonically across the bands, which is the property a
triage policy needs. The Élevé band is deliberately narrow (about 3% of
buildings). On this signal very few buildings can be confidently placed at
twice the average risk, and its observed rate (≈1.8×) is a little below its
predicted rate. A narrow escalation queue fits a scarce resource like senior
underwriters, but its stated probabilities should be read as "high, likely
somewhat overstated".""")

code("""order = np.argsort(-p, kind="stable"); ys = y[order]
frac = np.arange(1, len(ys) + 1) / len(ys); gains = np.cumsum(ys) / ys.sum()
plt.figure(figsize=(6.5, 5))
plt.plot(frac, gains, color="#2c7fb8", label="model")
plt.plot([0, 1], [0, 1], "--", color="grey", label="random")
plt.plot([0, y.mean(), 1], [0, 1, 1], ":", color="#27ae60", label="perfect")
plt.scatter([0.3], [tm.metrics["gains_top30"]], color="#c0392b", zorder=5)
plt.xlabel("fraction of portfolio reviewed (highest score first)")
plt.ylabel("fraction of TRUE claims captured"); plt.title("Cumulative gains (out-of-fold)")
plt.legend(); plt.tight_layout(); plt.show()
print(f"Reviewing the top 30% surfaces {tm.metrics['gains_top30']:.1%} of genuine claims "
      f"(random inspection: 30%) — a lift of {tm.metrics['gains_top30'] / 0.30:.2f}x.")""")

# --------------------------------------------------------------------------- 8
md("""## 8. Deployment & train/serve parity

Here the notebook loads the **deployed bundle** in `models/`, the exact
artefact the FastAPI service and the Streamlit app load, and scores the test
portfolio through `score_frame`. It then checks two things:

1. the model retrained in this notebook and the deployed bundle agree;
2. this output matches `predictions_test.csv` written by `train.py`.

The result is saved to `notebooks/notebook_predictions.csv`;
`tests/test_parity.py` sends the same raw rows to the running API and requires
identical answers.

*Why this section exists:* the earlier version stored Geo codes as float keys
that became strings after the JSON round-trip, so the deployed API silently
ignored location and **disagreed with the published predictions on 710 of
2,147 buildings**. No test compared the two paths. Now one does.""")

code("""deployed = TrainedModel.load(os.path.join(ROOT, "models"))
scored = deployed.score_frame(test)
scored.insert(0, "customer_id", test["Customer Id"].to_numpy())
scored.to_csv("notebook_predictions.csv", index=False)

in_notebook = tm.score_frame(test)
published = pd.read_csv(os.path.join(ROOT, "predictions_test.csv"))
print("max |p| difference, notebook-trained vs deployed bundle :",
      float(np.abs(in_notebook["claim_probability"] - scored["claim_probability"]).max()))
print("max |p| difference, deployed bundle vs train.py output  :",
      float(np.abs(published["claim_probability"] - scored["claim_probability"]).max()))
print("decision mismatches vs train.py output                 :",
      int((published["prediction"] != scored["prediction"]).sum()))
print("\\nTest predictions:", scored["prediction"].value_counts().to_dict())
print("Risk bands      :", scored["risk_level"].value_counts().reindex(BAND_NAMES).to_dict())
scored.head(10)""")

md("""### Serving the model

```bash
python train.py                               # trains, evaluates, writes models/ + predictions
uvicorn api.main:app --reload --port 8000     # REST API -> http://localhost:8000/docs
streamlit run app/streamlit_app.py            # interactive app
pytest -q                                     # full test suite, including parity
```""")

# --------------------------------------------------------------------------- 9
md("""## 9. Conclusions and limitations

**What this project shows** (figures quoted for the generated dataset; the
live cells above recompute them for whatever data is in `data/`)
- **Leakage, measured and removed.** Target encoding fitted before the split
  inflated AUC from ≈0.67 to ≈0.83, *above the generator's theoretical
  ceiling*. The fold-safe pipeline sits within ~0.02 of that ceiling.
- **Calibrated probabilities.** SMOTE-trained scores overstated risk about 2×.
  Cross-fitted Platt scaling brings the mean prediction to the observed rate
  and cuts expected calibration error from ≈0.25 to <0.01.
- **A decision, not a metric.** An explicit recall guarantee, bands tied to
  the portfolio claim rate with their *observed* rates reported, and a stated
  review-cost trade-off.
- **Train/serve parity under test.** Notebook, `train.py`, API and app share
  one scoring method, and the test suite checks that they agree.

**Limitations, stated plainly**
- Unless `data_source` above reads *real*, every number describes a generated
  dataset that mimics the competition schema. The method transfers; the
  magnitudes do not.
- The signal is weak (AUC ≈0.67). Holding 70% recall means reviewing about 60%
  of the portfolio, and the top band is slightly over-predicted.
- Geo_Code adds nothing at ≈4.5 buildings per zone. A hierarchical encoding
  (zone → region) or external geographic data would be the next lever.
- Thresholds optimise F1 under a recall constraint. With claim **severity**
  data they should minimise expected monetary loss instead.""")

nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Insurance_Claim_Risk_Case_Study.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")

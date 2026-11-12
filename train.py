"""
Training entrypoint for the Building Insurance Claim-Risk case study.

Runs the full pipeline and writes:
  * models/                       deployable bundle (model, scaler, calibrator,
                                  preprocessing state, metadata incl. data source)
  * models/leaderboard.csv        model comparison (held-out split)
  * models/oof_predictions.csv    out-of-fold raw + calibrated scores with labels
  * models/figures/*.png          EDA, ROC/PR, confusion, reliability, business
  * predictions_test.csv          scored test portfolio (via TrainedModel.score_frame)

Usage:
    python train.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data.make_dataset import generate, dataset_provenance       # noqa: E402
from diagnostics import leakage_study                             # noqa: E402
from src_pipeline import (train_full, read_raw_csv, TrainedModel,  # noqa: E402
                          MODEL_FEATURES)

MODEL_DIR = os.path.join(HERE, "models")
FIG_DIR = os.path.join(MODEL_DIR, "figures")
PALETTE = {"pos": "#c0392b", "neg": "#2c7fb8", "ok": "#27ae60", "mid": "#f39c12"}


def _footer(fig, source: str) -> None:
    """Stamp every figure with where its numbers come from."""
    if source == "generated":
        text = ("Data: reproducible GENERATED dataset matching the competition schema "
                "— not the real portfolio (see README › About the data)")
    else:
        text = "Data: real competition dataset"
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=8.5,
             color="#7f1d1d" if source == "generated" else "#555", style="italic")


def _save(fig, name: str, source: str) -> None:
    _footer(fig, source)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_eda(train: pd.DataFrame, source: str) -> None:
    y = (train["Claim"] == "oui").astype(int)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    vc = train["Claim"].value_counts()
    ax[0, 0].bar(vc.index, vc.values, color=[PALETTE["neg"], PALETTE["pos"]])
    ax[0, 0].set_title("Class balance (Claim)")
    for i, v in enumerate(vc.values):
        ax[0, 0].text(i, v, f"{v/len(train)*100:.1f}%", ha="center", va="bottom")
    ax[0, 1].hist(pd.to_numeric(train["Building Dimension"], errors="coerce").dropna(),
                  bins=60, color=PALETTE["neg"])
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Building Dimension (right-skewed, outliers)")
    rate = train.assign(y=y).groupby("Building_Type")["y"].mean().sort_values()
    ax[0, 2].barh(rate.index, rate.values, color=PALETTE["pos"])
    ax[0, 2].set_title("Claim rate by Building_Type")
    r2 = train.assign(y=y).groupby("Settlement")["y"].mean()
    ax[1, 0].bar(r2.index, r2.values, color=PALETTE["neg"])
    ax[1, 0].set_title("Claim rate by Settlement (U/R)")
    order = ["without"] + [str(i) for i in range(1, 10)] + [">=10"]
    wr = train.assign(y=y).groupby("NumberOfWindows")["y"].mean().reindex(order)
    ax[1, 1].bar(range(len(wr)), wr.values, color=PALETTE["pos"])
    ax[1, 1].set_xticks(range(len(wr)))
    ax[1, 1].set_xticklabels(wr.index, rotation=45, fontsize=7)
    ax[1, 1].set_title("Claim rate by NumberOfWindows")
    miss = train.isna().mean()
    miss = miss[miss > 0].sort_values()
    ax[1, 2].barh(miss.index, miss.values * 100, color=PALETTE["neg"])
    ax[1, 2].set_title("Missing values (%)")
    fig.tight_layout()
    _save(fig, "eda.png", source)


def fig_roc_pr(oof: pd.DataFrame, source: str) -> None:
    from sklearn.metrics import (roc_curve, precision_recall_curve,
                                 roc_auc_score, average_precision_score)
    y, p = oof["y"].to_numpy(), oof["p_calibrated"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, color=PALETTE["pos"], label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC (out-of-fold, all training rows)"); ax[0].legend()
    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec, color=PALETTE["neg"],
               label=f"AP = {average_precision_score(y, p):.3f}")
    ax[1].axhline(y.mean(), ls="--", color="grey", label=f"base rate {y.mean():.2f}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision–Recall (out-of-fold)"); ax[1].legend()
    fig.tight_layout()
    _save(fig, "roc_pr.png", source)


def fig_confusion_importance(tm: TrainedModel, oof: pd.DataFrame, source: str) -> None:
    from sklearn.metrics import confusion_matrix
    y = oof["y"].to_numpy()
    pred = (oof["p_calibrated"].to_numpy() >= tm.threshold).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(["Non", "Oui"]); ax[0].set_yticklabels(["Non", "Oui"])
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
    ax[0].set_title(f"Confusion matrix (out-of-fold) @ threshold {tm.threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    model = tm.model
    if hasattr(model, "coef_"):
        imp, label = model.coef_.ravel(), "Standardised coefficient (+ = more risk)"
    elif hasattr(model, "feature_importances_"):
        imp, label = np.asarray(model.feature_importances_), "Feature importance"
    else:
        imp, label = np.zeros(len(MODEL_FEATURES)), "n/a"
    order = np.argsort(np.abs(imp))
    colors = [PALETTE["pos"] if v > 0 else PALETTE["neg"] for v in imp[order]]
    ax[1].barh(np.array(MODEL_FEATURES)[order], imp[order], color=colors)
    ax[1].axvline(0, color="grey", lw=0.8)
    ax[1].set_title(label)
    fig.tight_layout()
    _save(fig, "confusion_importance.png", source)


def fig_calibration(tm: TrainedModel, oof: pd.DataFrame, source: str) -> None:
    """Reliability diagram: raw SMOTE-trained scores vs both calibrators."""
    y = oof["y"].to_numpy()
    cal = tm.metrics["calibration"]
    comp = cal["comparison"]
    chosen = cal["method"]
    bins = np.linspace(0, 1, 11)
    series = [("p_raw", PALETTE["pos"], "Raw (trained on SMOTE 50/50)",
               cal["ece_raw"], cal["brier_raw"], "-")]
    for method, color in [("sigmoid", PALETTE["neg"]), ("isotonic", PALETTE["ok"])]:
        tag = " ← deployed" if method == chosen else ""
        name = ("Platt / sigmoid" if method == "sigmoid" else "Isotonic") + tag
        series.append((f"p_{method}", color, name, comp[method]["ece"],
                       comp[method]["brier"], "-" if method == chosen else "--"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for col, color, name, ece, brier, ls in series:
        p = oof[col].to_numpy()
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 10:
                xs.append(p[mask].mean()); ys.append(y[mask].mean())
        ax[0].plot(xs, ys, "o", ls=ls, color=color,
                   label=f"{name}  ECE={ece:.3f}  Brier={brier:.3f}")
        if col in ("p_raw", f"p_{chosen}"):
            ax[1].hist(p, bins=40, range=(0, 1), alpha=0.55, color=color, label=name)
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    for e in tm.band_edges:
        ax[0].axvline(e, color="#999", lw=0.8, ls=":")
    ax[0].text(tm.band_edges[0], 0.97, " band edges", fontsize=7, color="#777", va="top")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("Mean predicted claim probability (bin)")
    ax[0].set_ylabel("Observed claim rate (bin)")
    ax[0].set_title("Reliability diagram (out-of-fold)")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[1].axvline(tm.metrics["base_rate"], color="black", ls="--",
                  label=f"true base rate {tm.metrics['base_rate']:.2f}")
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Buildings")
    ax[1].set_title("Score distribution: raw vs calibrated")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "calibration.png", source)


def fig_business(tm: TrainedModel, scored: pd.DataFrame, oof: pd.DataFrame,
                 source: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bc = scored["risk_level"].value_counts().reindex(["Faible", "Moyen", "Élevé"]).fillna(0)
    ax[0].bar(bc.index, bc.values, color=[PALETTE["ok"], PALETTE["mid"], PALETTE["pos"]])
    ax[0].set_title("Test portfolio triage: buildings per risk band")
    for i, v in enumerate(bc.values):
        ax[0].text(i, v, f"{int(v)}\n({v/len(scored)*100:.0f}%)", ha="center", va="bottom")
    ax[0].set_ylim(0, bc.max() * 1.18)

    y = oof["y"].to_numpy()
    order = np.argsort(-oof["p_calibrated"].to_numpy(), kind="stable")
    ys = y[order]
    frac = np.arange(1, len(ys) + 1) / len(ys)
    gains = np.cumsum(ys) / max(1, ys.sum())
    base = ys.mean()
    ax[1].plot(frac, gains, color=PALETTE["neg"], label="model")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax[1].plot([0, base, 1], [0, 1, 1], ":", color=PALETTE["ok"], label="perfect")
    at30 = tm.metrics["gains_top30"]
    ax[1].scatter([0.30], [at30], color=PALETTE["pos"], zorder=5)
    ax[1].annotate(f"review top 30% → {at30:.0%} of real claims",
                   xy=(0.30, at30), xytext=(0.38, max(0.12, at30 - 0.25)), fontsize=9,
                   arrowprops=dict(arrowstyle="->", color=PALETTE["pos"]))
    ax[1].set_xlabel("Fraction of portfolio reviewed (highest score first)")
    ax[1].set_ylabel("Fraction of TRUE claims captured")
    ax[1].set_title("Cumulative gains (out-of-fold, true labels)")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "business.png", source)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    train_csv, test_csv = generate()
    source = dataset_provenance(train_csv)
    print(f"Data source: {source.upper()}")

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)

    print("Training pipeline...")
    tm = train_full(train_csv, verbose=True, data_source=source)
    tm.save(MODEL_DIR)
    pd.DataFrame(tm.metrics["leaderboard"]).to_csv(
        os.path.join(MODEL_DIR, "leaderboard.csv"), index=False)
    tm.oof.to_csv(os.path.join(MODEL_DIR, "oof_predictions.csv"), index=False)

    m = tm.metrics
    print(f"\nChampion: {m['champion']}")
    print(f"Decision threshold (calibrated scale, recall>=0.70): {tm.threshold:.3f}")
    print("Out-of-fold @ threshold:", json.dumps(
        {k: round(v, 4) for k, v in m["oof_at_threshold"].items()}))
    print(f"CV F1: {m['cv']['f1_mean']:.4f} ± {m['cv']['f1_std']:.4f}")
    c = m["calibration"]
    for k, v in c["comparison"].items():
        print(f"  calibrator {k:8s} log-loss={v['log_loss']:.4f} "
              f"Brier={v['brier']:.4f} ECE={v['ece']:.4f}")
    print(f"Deployed calibrator: {c['method']}")
    print(f"Calibration  Brier {c['brier_raw']:.4f} -> {c['brier_calibrated']:.4f} | "
          f"ECE {c['ece_raw']:.4f} -> {c['ece_calibrated']:.4f} | "
          f"mean score {c['mean_pred_raw']:.3f} -> {c['mean_pred_calibrated']:.3f} "
          f"(base rate {c['base_rate']:.3f})")
    print(f"Top-30% review captures {m['gains_top30']:.1%} of real claims")
    print(f"Threshold flags {m['flagged_share']:.1%} of the portfolio")
    for b in m["bands"]:
        print(f"  band {b['band']:7s} n={b['n']:5d} share={b['share']:.1%} "
              f"predicted={b['mean_predicted']:.3f} observed={b['observed_rate']:.3f}")

    # Score the test portfolio through the SAME method the API uses.
    scored = tm.score_frame(test)
    scored.insert(0, "customer_id", test["Customer Id"].to_numpy())
    scored.to_csv(os.path.join(HERE, "predictions_test.csv"), index=False)

    print("\nLeakage study (5-fold CV AUC)...")
    study = leakage_study(train_csv)
    with open(os.path.join(MODEL_DIR, "leakage_study.json"), "w") as f:
        json.dump(study, f, indent=2)
    for k, v in study.items():
        print(f"  {k:32s} {v if isinstance(v, str) or v is None else round(v, 4)}")

    fig_eda(train, source)
    fig_roc_pr(tm.oof, source)
    fig_confusion_importance(tm, tm.oof, source)
    fig_calibration(tm, tm.oof, source)
    fig_business(tm, scored, tm.oof, source)

    print("\nTest predictions:", scored["prediction"].value_counts().to_dict())
    print("Risk bands:", scored["risk_level"].value_counts().to_dict())
    print("Artifacts written to models/ and predictions_test.csv")


if __name__ == "__main__":
    main()

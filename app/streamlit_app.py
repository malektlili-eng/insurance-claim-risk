"""
Streamlit app — Insurance Claim-Risk Scorer.
================================================================================

Run with:
    streamlit run app/streamlit_app.py

Three tabs:
  1. Score a building   -> real-time single-record prediction
  2. Batch scoring      -> upload a CSV, get a scored portfolio + download
  3. Model card         -> data provenance, calibration, bands, leaderboard

All scoring goes through `TrainedModel.score_frame`, the same method the API,
train.py and the notebook use.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src_pipeline import TrainedModel, TRIAGE_ACTION, BAND_NAMES, CSV_DTYPES  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")
BAND_COLOR = {"Faible": "#27ae60", "Moyen": "#f39c12", "Élevé": "#c0392b"}

st.set_page_config(page_title="Insurance Claim-Risk Scorer", page_icon="🏠",
                   layout="wide")


@st.cache_resource
def load_model() -> TrainedModel | None:
    if not os.path.exists(os.path.join(MODEL_DIR, "model.joblib")):
        return None
    return TrainedModel.load(MODEL_DIR)


def risk_gauge(prob: float, band: str, base_rate: float) -> str:
    color = BAND_COLOR[band]
    pct = min(100, int(prob * 100))
    rel = prob / base_rate if base_rate else 0
    return f"""
    <div style="border:1px solid #ddd;border-radius:12px;padding:18px;background:#fafafa;">
      <div style="font-size:13px;color:#666;">Calibrated claim probability</div>
      <div style="font-size:44px;font-weight:700;color:{color};">{prob:.1%}</div>
      <div style="background:#eee;border-radius:8px;height:14px;overflow:hidden;margin-top:8px;">
        <div style="width:{pct}%;height:100%;background:{color};"></div>
      </div>
      <div style="margin-top:12px;font-size:15px;">
        Risk band: <b style="color:{color};">{band}</b>
        &nbsp;·&nbsp; {rel:.1f}× the portfolio average ({base_rate:.1%})
      </div>
    </div>
    """


model = load_model()

st.title("🏠 Insurance Claim-Risk Scorer")
st.caption("Underwriting-triage tool for building insurance — calibrated claim "
           "probability, risk banding and recommended action.")

if model is None:
    st.error("No trained model found. Run `python train.py` from the project root, "
             "then reload this app.")
    st.stop()

if model.data_source == "generated":
    st.warning("**Demo model.** It was trained on a reproducible *generated* dataset "
               "that matches the competition schema, not on a real insurance "
               "portfolio. Scores illustrate the system; they are not real risk "
               "estimates. See the README › *About the data*.")

base_rate = model.metrics.get("base_rate", 0.22)
tab1, tab2, tab3 = st.tabs(["🎯 Score a building", "📦 Batch scoring", "📊 Model card"])

# --------------------------------------------------------------------------- #
# Tab 1 — single prediction
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Score a single building")
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.selectbox("Year of observation", [2012, 2013, 2014, 2015, 2016], index=2)
        period = st.selectbox("Insured period (years)", [0.5, 1.0], index=1)
        residential = st.radio("Residential?", [0, 1], index=1, horizontal=True,
                               format_func=lambda x: "Yes" if x else "No")
        btype = st.selectbox("Building type", ["Fire-resistive", "Non-combustible",
                                               "Ordinary", "Wood-framed"], index=2)
    with c2:
        painted = st.radio("Painted?", ["V", "N"], index=0, horizontal=True,
                           format_func=lambda x: "Yes" if x == "V" else "No")
        fenced = st.radio("Fenced?", ["V", "N"], index=1, horizontal=True,
                          format_func=lambda x: "Yes" if x == "V" else "No")
        garden = st.radio("Garden?", ["V", "O"], index=0, horizontal=True,
                          format_func=lambda x: "Yes" if x == "V" else "No")
        settlement = st.radio("Settlement", ["U", "R"], index=0, horizontal=True,
                              format_func=lambda x: "Urban" if x == "U" else "Rural")
    with c3:
        dim = st.number_input("Building dimension (m²)", min_value=50.0,
                              max_value=25000.0, value=1200.0, step=50.0)
        windows = st.selectbox("Number of windows", ["without", "1", "2", "3", "4", "5",
                                                     "6", "7", "8", "9", ">=10"], index=4)
        geo = st.text_input("Geo code (leave blank if unknown)", value="0042")

    if st.button("Score building", type="primary"):
        record = {
            "YearOfObservation": year, "Insured_Period": period,
            "Residential": residential, "Building_Painted": painted,
            "Building_Fenced": fenced, "Garden": garden, "Settlement": settlement,
            "Building Dimension": dim, "Building_Type": btype,
            "NumberOfWindows": windows, "Geo_Code": geo.strip() or None,
        }
        res = model.score_record(record)
        g1, g2 = st.columns([1, 1.3])
        with g1:
            st.markdown(risk_gauge(res["claim_probability"], res["risk_level"], base_rate),
                        unsafe_allow_html=True)
        with g2:
            st.metric(f"Decision (threshold {res['threshold_used']:.3f})",
                      "Flag for review" if res["prediction"] == "oui" else "No flag")
            st.info(f"**Recommended action:** {res['recommended_action']}")
            st.caption("The threshold keeps recall on true claims ≥ 70%: in claims "
                       "triage, missing a real claim costs more than a false alarm.")

# --------------------------------------------------------------------------- #
# Tab 2 — batch scoring
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Score a portfolio from CSV")
    st.caption("Upload a CSV with the raw dataset columns. A `Customer Id` column is "
               "optional and kept in the output. Blank values are imputed.")
    up = st.file_uploader("CSV file", type=["csv"])
    demo = st.checkbox("Use the bundled test set instead")

    df = None
    if demo:
        path = os.path.join(ROOT, "data", "test_Insurance.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=CSV_DTYPES)
    elif up is not None:
        df = pd.read_csv(up, dtype=CSV_DTYPES)

    if df is not None:
        try:
            out = model.score_frame(df)
            if "Customer Id" in df.columns:
                out.insert(0, "customer_id", df["Customer Id"].to_numpy())

            m1, m2, m3 = st.columns(3)
            m1.metric("Buildings scored", len(out))
            m2.metric("Flagged for review", int((out["prediction"] == "oui").sum()))
            m3.metric("Escalations (Élevé)", int((out["risk_level"] == "Élevé").sum()))
            st.bar_chart(out["risk_level"].value_counts().reindex(BAND_NAMES).fillna(0))
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Download scored CSV", out.to_csv(index=False).encode(),
                               "scored_portfolio.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not score this file: {e}")

# --------------------------------------------------------------------------- #
# Tab 3 — model card
# --------------------------------------------------------------------------- #
with tab3:
    met = model.metrics
    st.subheader("Model card")
    st.markdown(f"**Training data:** `{model.data_source}` · "
                f"{met.get('n_train_rows', '?')} buildings · "
                f"portfolio claim rate {base_rate:.1%}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Champion", met.get("champion", "—"))
    k2.metric("CV AUC (5-fold)", f"{met['cv']['auc_mean']:.3f}")
    k3.metric("Recall @ threshold", f"{met['oof_at_threshold']['recall']:.0%}")
    k4.metric("Portfolio flagged", f"{met.get('flagged_share', 0):.0%}")

    cal = met.get("calibration", {})
    if cal:
        st.markdown(f"**Calibration** — deployed method: `{cal['method']}` "
                    f"(selected by {cal['selection']})")
        st.dataframe(pd.DataFrame({
            "Scores": ["Raw (SMOTE-trained)"] + [f"Calibrated · {k}" for k in cal["comparison"]],
            "Brier": [cal["brier_raw"]] + [v["brier"] for v in cal["comparison"].values()],
            "ECE": [cal["ece_raw"]] + [v["ece"] for v in cal["comparison"].values()],
            "Log-loss": [cal["log_loss_raw"]] + [v["log_loss"] for v in cal["comparison"].values()],
        }).round(4), use_container_width=True, hide_index=True)

    st.markdown("**Triage policy** (out-of-fold, every training building held out once)")
    bands = pd.DataFrame(met.get("bands", []))
    if not bands.empty:
        lo, hi = model.band_edges
        bands["probability range"] = [f"< {lo:.3f}", f"{lo:.3f} – {hi:.3f}", f"≥ {hi:.3f}"]
        bands["action"] = bands["band"].map(TRIAGE_ACTION)
        bands["share"] = (bands["share"] * 100).round(1).astype(str) + "%"
        st.dataframe(bands[["band", "probability range", "share", "mean_predicted",
                            "observed_rate", "action"]].round(3),
                     use_container_width=True, hide_index=True)

    lb_path = os.path.join(MODEL_DIR, "leaderboard.csv")
    if os.path.exists(lb_path):
        st.markdown("**Model leaderboard** (held-out split, raw scores @ 0.5)")
        st.dataframe(pd.read_csv(lb_path).round(4), use_container_width=True,
                     hide_index=True)

    fig_dir = os.path.join(MODEL_DIR, "figures")
    for name, cap in [("calibration.png", "Reliability diagram"),
                      ("roc_pr.png", "ROC & Precision–Recall"),
                      ("business.png", "Portfolio triage & cumulative gains")]:
        path = os.path.join(fig_dir, name)
        if os.path.exists(path):
            st.image(path, caption=cap, use_container_width=True)

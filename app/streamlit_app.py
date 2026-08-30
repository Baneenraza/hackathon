"""Block 11 - AI Factory Intelligence Command Center (Streamlit).

Per-modality inputs -> 4-agent orchestration -> prediction + confidence +
explanation (SHAP / Grad-CAM) -> digital-twin what-if -> human APPROVE / REJECT /
MODIFY (logged to CSV) -> downloadable fpdf2 incident report.

The AI is decision support only. Every recommendation needs a named human.
"""
import os
import sys
import uuid
from datetime import datetime

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# stability on small shared containers (avoids OpenMP oversubscription / oneDNN crashes)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C
from agents.orchestrator import Orchestrator
from report.pdf_report import build_report
from xai.shap_tabular import explain_instance

DECISION_LOG = os.path.join(C.PROCESSED, "decisions_log.csv")
CACHE = os.path.join(C.REPORTS, "app_cache")
os.makedirs(CACHE, exist_ok=True)

st.set_page_config(page_title="AI Factory Command Center", layout="wide")


@st.cache_resource(show_spinner="Loading models (one-time)...")
def get_orchestrator():
    return Orchestrator(lazy=True)


@st.cache_data
def ai4i_test():
    return pd.read_csv(C.TABULAR_CSV)


@st.cache_data
def cmapss_test():
    df = pd.read_csv(C.CMAPSS_TEST, sep=r"\s+", header=None, names=C.CMAPSS_COLS)
    import json
    cols = json.load(open(os.path.join(C.MODELS, "lstm_rul_config.json")))["sensor_cols"]
    return df, cols


def gradcam_overlay(agent, image_path):
    out = agent.run(image_path, return_cam=True)
    cam = out.pop("cam")
    from tensorflow.keras.utils import load_img, img_to_array
    img = img_to_array(load_img(image_path, target_size=(agent.img, agent.img))) / 255
    import tensorflow as tf
    cam_rs = tf.image.resize(cam[..., None], (agent.img, agent.img)).numpy().squeeze()
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(img); ax.imshow(cam_rs, cmap="jet", alpha=0.45); ax.axis("off")
    ax.set_title(f"Grad-CAM  p(defect)={out['p_defect']:.2f}", fontsize=9)
    p = os.path.join(CACHE, "gradcam.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return out, p


def shap_bar(contribs):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    names = [c["friendly"] for c in contribs][::-1]
    vals = [c["shap"] for c in contribs][::-1]
    ax.barh(names, vals, color=["#c44e52" if v > 0 else "#4c72b0" for v in vals])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP contribution to failure probability")
    p = os.path.join(CACHE, "shap.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def log_decision(row):
    df = pd.DataFrame([row])
    df.to_csv(DECISION_LOG, mode="a", header=not os.path.exists(DECISION_LOG),
              index=False)


# ----------------------------- UI -----------------------------
st.title("AI Factory Intelligence Command Center")
st.caption("Unified view over an AI4I predictive-maintenance line, a CMAPSS "
           "turbofan fleet and a casting-QA station.")

st.warning("**Decision support only.** Every prediction is probabilistic and "
           "advisory. A named human holds sole authority for any action. "
           "Knowledge-base documents in this demo are synthetic.")

orch = get_orchestrator()
ss = st.session_state

with st.sidebar:
    st.header("Incident inputs")
    asset = st.text_input("Machine / asset ID", "CNC-L-02")
    mode = st.selectbox("Failure-mode hint (drives the KB query)",
                        ["", "TWF", "HDF", "PWF", "OSF", "RNF"], index=4)
    custom_q = st.text_input("or custom knowledge query", "")

    st.subheader("1. Process sensors (tabular)")
    use_tab = st.checkbox("Include tabular failure prediction", True)
    if st.button("Load random AI4I test row"):
        r = ai4i_test().sample(1).iloc[0]
        ss.tab = {"air_temp_K": float(r["Air temperature [K]"]),
                  "process_temp_K": float(r["Process temperature [K]"]),
                  "rot_speed_rpm": float(r["Rotational speed [rpm]"]),
                  "torque_Nm": float(r["Torque [Nm]"]),
                  "tool_wear_min": float(r["Tool wear [min]"]),
                  "type": r["Type"]}
    tab = ss.get("tab", {"air_temp_K": 300.0, "process_temp_K": 310.0,
                         "rot_speed_rpm": 1500.0, "torque_Nm": 40.0,
                         "tool_wear_min": 108.0, "type": "L"})
    c1, c2 = st.columns(2)
    tab["air_temp_K"] = c1.number_input("Air temp [K]", 295.0, 305.0,
                                        float(tab["air_temp_K"]), 0.1)
    tab["process_temp_K"] = c2.number_input("Process temp [K]", 305.0, 315.0,
                                            float(tab["process_temp_K"]), 0.1)
    tab["rot_speed_rpm"] = c1.number_input("Rot. speed [rpm]", 1160.0, 2890.0,
                                           float(tab["rot_speed_rpm"]), 10.0)
    tab["torque_Nm"] = c2.number_input("Torque [Nm]", 3.0, 80.0,
                                       float(tab["torque_Nm"]), 0.5)
    tab["tool_wear_min"] = c1.number_input("Tool wear [min]", 0.0, 260.0,
                                           float(tab["tool_wear_min"]), 1.0)
    tab["type"] = c2.selectbox("Type", ["L", "M", "H"],
                               ["L", "M", "H"].index(tab["type"]))
    ss.tab = tab

    st.subheader("2. Turbofan sensor history (CMAPSS)")
    use_ts = st.checkbox("Include LSTM RUL estimate", True)
    engine = st.number_input("Test engine #", 1, 100, 1)

    st.subheader("3. Casting image (vision)")
    up = st.file_uploader("Upload a casting photo", ["jpg", "jpeg", "png"])
    sample_choice = st.radio("or use a sample", ["none", "defect sample", "ok sample"],
                             horizontal=True)

run = st.button("Run full analysis", type="primary", use_container_width=True)

if run:
    image_path = None
    if up is not None:
        image_path = os.path.join(CACHE, "upload_" + up.name)
        open(image_path, "wb").write(up.getbuffer())
    elif sample_choice != "none":
        import glob
        sub = "def_front" if "defect" in sample_choice else "ok_front"
        pool = (glob.glob(os.path.join(C.ROOT, "data", "sample_images", sub, "*.jpeg"))
                or glob.glob(os.path.join(C.CASTING_DIR, "test", sub, "*.jpeg")))
        image_path = sorted(pool)[int(engine) % len(pool)]

    sw = None
    if use_tab or use_ts:
        sw = {}
        if use_tab:
            sw["tabular"] = ss.tab
        if use_ts:
            df, cols = cmapss_test()
            g = df[df.unit == int(engine)].sort_values("cycle")
            sw["cmapss_window"] = g[cols].values[-50:].tolist()

    q = custom_q or None
    with st.spinner("Running VisionAgent -> PredictiveAgent -> KnowledgeAgent -> "
                    "PlanningAgent ..."):
        result = orch.run(image=image_path, sensor_window=sw, query=q,
                          failure_mode_hint=mode or None)

    # explanations
    shap_out, shap_png = None, None
    if use_tab:
        shap_out = explain_instance(ss.tab)
        shap_png = shap_bar(shap_out["contributions"])
    gradcam_png = None
    if image_path:
        vout, gradcam_png = gradcam_overlay(orch.vision, image_path)
        result["vision"] = {**result["vision"], **vout}

    ss.result = result
    ss.shap_out = shap_out
    ss.shap_png = shap_png
    ss.gradcam_png = gradcam_png
    ss.image_path = image_path
    ss.asset = asset
    ss.mode = mode
    ss.report_id = "INC-" + datetime.now().strftime("%Y%m%d-%H%M%S")

# ----------------------------- results -----------------------------
if ss.get("result"):
    r = ss.result
    dec = r["decision"]
    st.divider()
    st.subheader(f"Results  ·  {ss.report_id}  ·  {ss.asset}")
    st.caption("Agent chain: " + "  ->  ".join(r["agent_trace"])
               + f"   ({r['elapsed_s']} s)")

    colv, colp, colk = st.columns(3)
    with colv:
        st.markdown("### VisionAgent")
        v = r["vision"]
        if v.get("skipped"):
            st.info("no image provided")
        else:
            st.metric("Defect", "YES" if v["defect"] else "NO")
            st.write(f"severity **{v['severity']:.0%}**, confidence "
                     f"**{v['confidence']:.0%}**")
            if ss.gradcam_png:
                st.image(ss.gradcam_png, caption="Grad-CAM: pixels driving the call")
    with colp:
        st.markdown("### PredictiveAgent")
        p = r["predictive"]
        st.metric("Failure probability", f"{p['failure_prob']:.0%}")
        st.metric("RUL estimate", f"{p['rul_estimate']:.0f} h")
        st.write(f"confidence **{p['confidence']:.0%}**")
        if ss.get("shap_out"):
            st.caption("Why (SHAP): " + ss.shap_out["why"])
            st.image(ss.shap_png)
    with colk:
        st.markdown("### KnowledgeAgent")
        k = r["knowledge"]
        if k.get("skipped"):
            st.info("no query")
        else:
            st.write(k["answer"])
            if k.get("source_doc"):
                st.success(f"Source: **{k['source_doc']}** section "
                           f"{k['source_section']} - {k.get('source_heading')}")
            with st.expander("retrieved passages"):
                st.table(pd.DataFrame(k["retrieved"]))

    st.markdown("### PlanningAgent - recommendation")
    lvl = dec["risk_level"]
    {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.info}[lvl](
        f"**Risk {lvl}** (score {dec['risk_score']})  ·  {dec['recommendation']}")

    sc = dec["simulated_outcomes"]
    tw = pd.DataFrame(sc["scenarios"])
    st.markdown(f"#### Digital-twin what-if  ·  horizon {sc['horizon_hours']:.0f} h "
                f"·  twin picks **{sc['twin_recommended_scenario']}**")
    st.caption("Driven live by the failure probability + RUL above - no hardcoded "
               "outcomes.")
    cc1, cc2 = st.columns([1.3, 1])
    cc1.dataframe(tw.set_index("scenario"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.bar(tw["scenario"], tw["expected_units"], color="#4c72b0")
    ax2 = ax.twinx()
    ax2.plot(tw["scenario"], tw["risk_score"], "o-", color="#c44e52")
    ax.set_ylabel("expected units"); ax2.set_ylabel("risk score", color="#c44e52")
    fig.tight_layout(); cc2.pyplot(fig)

    # ----------------------------- decision panel -----------------------------
    st.divider()
    st.subheader("Human decision  (required)")
    d1, d2 = st.columns([1, 2])
    choice = d1.radio("Decision", ["APPROVE", "REJECT", "MODIFY"])
    who = d1.text_input("Decided by", "shift engineer")
    reason = d2.text_area("Reason / modification notes",
                          "Reviewed against SOP; ", height=110)

    if d1.button("Log decision", type="primary"):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "report_id": ss.report_id, "asset": ss.asset,
            "failure_mode_hint": ss.mode,
            "failure_prob": r["predictive"]["failure_prob"],
            "rul_estimate": r["predictive"]["rul_estimate"],
            "risk_level": lvl, "twin_recommended": sc["twin_recommended_scenario"],
            "human_decision": choice, "decided_by": who,
            "decision_reason": reason,
        }
        log_decision(row)
        ss.logged = row
        st.success(f"Logged to {os.path.relpath(DECISION_LOG, C.ROOT)}")

    if ss.get("logged"):
        payload = {
            "report_id": ss.report_id, "asset": ss.asset,
            "failure_mode_hint": ss.mode or "-",
            "timestamp": ss.logged["timestamp"],
            "orchestrator": r,
            "shap_why": (ss.shap_out or {}).get("why", ""),
            "shap_png": ss.shap_png, "gradcam_png": ss.gradcam_png,
            "human_decision": ss.logged["human_decision"],
            "decided_by": ss.logged["decided_by"],
            "decision_reason": ss.logged["decision_reason"],
        }
        pdf_path = os.path.join(C.REPORTS, f"{ss.report_id}.pdf")
        build_report(payload, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download incident & decision PDF", f,
                               file_name=f"{ss.report_id}.pdf",
                               mime="application/pdf", type="primary")

    with st.expander("decision log"):
        if os.path.exists(DECISION_LOG):
            st.dataframe(pd.read_csv(DECISION_LOG))
        else:
            st.write("no decisions logged yet")
else:
    st.info("Set inputs in the sidebar and press **Run full analysis**.")

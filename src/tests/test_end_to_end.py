"""Smoke test: exercise the exact path the Streamlit 'Run full analysis' button
runs, without a browser. Covers all 12 blocks' runtime artifacts.
"""
import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C
from agents.orchestrator import Orchestrator
from report.pdf_report import build_report
from xai.shap_tabular import explain_instance


def main():
    orch = Orchestrator(lazy=True)

    tab = {"air_temp_K": 302.0, "process_temp_K": 311.5, "rot_speed_rpm": 1330,
           "torque_Nm": 60, "tool_wear_min": 220, "type": "L"}
    df = pd.read_csv(C.CMAPSS_TEST, sep=r"\s+", header=None, names=C.CMAPSS_COLS)
    cols = json.load(open(os.path.join(C.MODELS, "lstm_rul_config.json")))["sensor_cols"]
    window = df[df.unit == 3].sort_values("cycle")[cols].values[-50:].tolist()
    img = sorted(glob.glob(os.path.join(C.CASTING_DIR, "test", "def_front", "*.jpeg")))[5]

    result = orch.run(image=img,
                      sensor_window={"tabular": tab, "cmapss_window": window},
                      failure_mode_hint="OSF")

    assert set(result["vision"]) >= {"defect", "severity", "confidence"}
    assert set(result["predictive"]) >= {"failure_prob", "rul_estimate", "confidence"}
    assert set(result["knowledge"]) >= {"answer", "source_doc", "source_section"}
    d = result["decision"]
    assert set(d) >= {"recommendation", "risk_level", "simulated_outcomes",
                      "requires_human_approval"}
    assert d["requires_human_approval"] is True
    assert d["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    scen = d["simulated_outcomes"]["scenarios"]
    assert {s["scenario"] for s in scen} == {"continue", "maintenance_stop",
                                             "reduced_load"}
    for s in scen:
        assert set(s) >= {"expected_units", "downtime_hours", "cost", "risk_score"}

    shap_out = explain_instance(tab)
    assert shap_out["contributions"] and shap_out["why"]

    payload = {"report_id": "TEST-0001", "asset": "CNC-L-02",
               "failure_mode_hint": "OSF",
               "timestamp": datetime.now().isoformat(timespec="seconds"),
               "orchestrator": result, "shap_why": shap_out["why"],
               "shap_png": None, "gradcam_png": None,
               "human_decision": "APPROVE", "decided_by": "tester",
               "decision_reason": "smoke test"}
    pdf = build_report(payload, os.path.join(C.REPORTS, "test_incident_report.pdf"))
    assert os.path.getsize(pdf) > 1000

    print("vision     :", result["vision"])
    print("predictive :", result["predictive"])
    print("knowledge  :", result["knowledge"]["source_doc"],
          result["knowledge"]["source_section"])
    print("decision   :", d["risk_level"], "|",
          d["simulated_outcomes"]["twin_recommended_scenario"])
    print("shap why   :", shap_out["why"])
    print("pdf        :", pdf, os.path.getsize(pdf), "bytes")
    print("\nALL BLOCKS E2E OK")


if __name__ == "__main__":
    main()

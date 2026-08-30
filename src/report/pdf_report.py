"""Block 11 - incident / decision PDF report (fpdf2).

build_report(payload, out_path) where payload holds the orchestrator output, the
SHAP 'why', the human decision + reason, and optional image paths (Grad-CAM,
SHAP, twin chart). Returns out_path.
"""
import os
from datetime import datetime

from fpdf import FPDF


def _clean(s):
    """fpdf2 core fonts are latin-1 only - drop anything outside it."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 8, "AI Factory Intelligence Command Center", ln=1)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, "Incident & Decision Report  -  DECISION SUPPORT ONLY, "
                        "not an autonomous action", ln=1)
        self.set_text_color(0, 0, 0)
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Generated {datetime.now():%Y-%m-%d %H:%M}  -  page "
                         f"{self.page_no()}", align="C")

    def h2(self, t):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(238, 240, 245)
        self.cell(0, 7, _clean("  " + t), ln=1, fill=True)
        self.ln(1)
        self.set_font("Helvetica", "", 10)

    def kv(self, k, v):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.cell(50, 6, _clean(k))
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin + 50)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 50, 6, _clean(v))

    def para(self, t):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, _clean(t))
        self.ln(1)

    def table(self, headers, rows):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        w = (self.w - self.l_margin - self.r_margin) / len(headers)
        for h in headers:
            self.cell(w, 7, _clean(h), border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        for r in rows:
            for c in r:
                self.cell(w, 6.5, _clean(c), border=1, align="C")
            self.ln()
        self.ln(2)


def build_report(payload, out_path):
    d = payload
    dec = d["orchestrator"]["decision"]
    pred = d["orchestrator"]["predictive"]
    vis = d["orchestrator"]["vision"]
    kno = d["orchestrator"]["knowledge"]

    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.h2("1. Incident summary")
    pdf.kv("Report ID", d.get("report_id", "-"))
    pdf.kv("Timestamp", d.get("timestamp", datetime.now().isoformat(timespec="seconds")))
    pdf.kv("Machine / asset", d.get("asset", "-"))
    pdf.kv("Failure mode hint", d.get("failure_mode_hint", "-"))
    pdf.kv("Overall risk level", dec["risk_level"] + f"  (score {dec['risk_score']})")

    pdf.h2("2. Model predictions")
    pdf.kv("Failure probability", f"{pred['failure_prob']:.1%}  "
                                  f"(confidence {pred['confidence']:.0%})")
    pdf.kv("RUL estimate", f"{pred['rul_estimate']:.0f} h")
    if not vis.get("skipped"):
        pdf.kv("Visual inspection", f"{'DEFECT' if vis['defect'] else 'OK'}  "
                                    f"(severity {vis['severity']:.0%}, "
                                    f"confidence {vis['confidence']:.0%})")
    pdf.kv("Prediction source", "tabular classifier (failure prob), LSTM (RUL), "
                                "MobileNetV2 (vision) - not the LLM")

    pdf.h2("3. Explanation (why)")
    if d.get("shap_why"):
        pdf.para(d["shap_why"])
    if d.get("shap_png") and os.path.exists(d["shap_png"]):
        pdf.image(d["shap_png"], w=150)
        pdf.ln(2)
    if d.get("gradcam_png") and os.path.exists(d["gradcam_png"]):
        pdf.para("Grad-CAM - image regions that drove the vision model:")
        pdf.image(d["gradcam_png"], w=110)
        pdf.ln(2)

    pdf.h2("4. Knowledge base guidance")
    if kno.get("answer"):
        pdf.para(kno["answer"])
        pdf.kv("Source", f"{kno.get('source_doc')} section {kno.get('source_section')}"
                         f" ({kno.get('source_heading')})")
    else:
        pdf.para("No knowledge-base query was run for this incident.")

    pdf.h2("5. Digital-twin what-if simulation")
    sc = dec["simulated_outcomes"]
    pdf.para(f"Planning horizon: {sc['horizon_hours']:.0f} h. Numbers are driven by "
             f"the failure probability and RUL above (not hardcoded).")
    rows = [[s["scenario"],
             f"{s['expected_units']:.0f}",
             f"{s['downtime_hours']:.1f}",
             f"{s['cost']:,.0f}",
             f"{s['risk_score']:.2f}"] for s in sc["scenarios"]]
    pdf.table(["scenario", "exp. units", "downtime h", "cost", "risk"], rows)
    pdf.kv("Twin-recommended", sc["twin_recommended_scenario"])

    pdf.h2("6. Recommendation (decision support)")
    pdf.para(dec["recommendation"])
    pdf.kv("Requires human approval", "YES - always")

    pdf.h2("7. Human decision")
    pdf.kv("Decision", d.get("human_decision", "-"))
    pdf.kv("Decided by", d.get("decided_by", "-"))
    pdf.kv("Reason / notes", d.get("decision_reason", "-"))

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, _clean(
        "This system provides decision support only. All predictions are "
        "probabilistic and were reviewed by a named human who holds sole "
        "authority for the action taken. Text/PDF knowledge sources in this "
        "deployment are synthetic."))

    pdf.output(out_path)
    return out_path


if __name__ == "__main__":
    demo = {
        "report_id": "INC-0001", "asset": "CNC cell L-02",
        "failure_mode_hint": "OSF",
        "orchestrator": {
            "vision": {"defect": True, "severity": 0.66, "confidence": 0.83},
            "predictive": {"failure_prob": 0.94, "rul_estimate": 120, "confidence": 0.49},
            "knowledge": {"answer": "Reduce feed rate, load below 50%.",
                          "source_doc": "safety_sop.pdf", "source_section": "3.0",
                          "source_heading": "Response to Overstrain"},
            "decision": {"risk_level": "HIGH", "risk_score": 0.61,
                         "recommendation": "Schedule a maintenance stop now.",
                         "simulated_outcomes": {"horizon_hours": 168,
                             "twin_recommended_scenario": "maintenance_stop",
                             "scenarios": [
                                 {"scenario": "continue", "expected_units": 9274,
                                  "downtime_hours": 13.4, "cost": 111331, "risk_score": 0.96},
                                 {"scenario": "maintenance_stop", "expected_units": 9840,
                                  "downtime_hours": 4, "cost": 77600, "risk_score": 0.02},
                                 {"scenario": "reduced_load", "expected_units": 5806,
                                  "downtime_hours": 6.7, "cost": 82545, "risk_score": 0.48}]}},
        },
        "shap_why": "Predicted FAILURE (94%). Main drivers: overstrain raises risk.",
        "human_decision": "APPROVE", "decided_by": "shift engineer",
        "decision_reason": "Agrees with SOP 3.0; stop scheduled for 22:00.",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    import sys
    sys.path.insert(0, "src")
    import config as C
    p = build_report(demo, os.path.join(C.REPORTS, "sample_incident_report.pdf"))
    print("wrote", p)

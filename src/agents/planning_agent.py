"""PlanningAgent - fuse the three agent outputs into a risk-rated recommendation.

run(vision_out, predictive_out, knowledge_out) -> {
    "recommendation": str,
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "simulated_outcomes": dict,          # 3 digital-twin scenarios + choice
    "requires_human_approval": True,
}

The digital-twin numbers are produced live by FactorySimulator from the REAL
failure_prob / rul_estimate in `predictive_out`. The agent never has autonomous
authority - `requires_human_approval` is always True.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digital_twin.simulator import FactorySimulator

RISK_W = {"failure_prob": 0.5, "rul": 0.3, "severity": 0.2}


class PlanningAgent:
    def __init__(self, horizon_hours=168.0):
        self.horizon = horizon_hours

    def _risk(self, failure_prob, rul_estimate, severity, rul_cap=125.0):
        rul_term = 1.0 - min(rul_estimate / rul_cap, 1.0)
        score = (RISK_W["failure_prob"] * failure_prob
                 + RISK_W["rul"] * rul_term
                 + RISK_W["severity"] * severity)
        level = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.3 else "LOW"
        return round(float(score), 4), level

    def run(self, vision_out, predictive_out, knowledge_out):
        fp = float(predictive_out.get("failure_prob", 0.0))
        rul = float(predictive_out.get("rul_estimate", 125.0))
        defect = bool(vision_out.get("defect", False))
        severity = float(vision_out.get("severity", 0.0)) if defect else 0.0

        risk_score, risk_level = self._risk(fp, rul, severity)

        sim = FactorySimulator(failure_prob=fp, rul_estimate=rul,
                               horizon_hours=self.horizon)
        best, df = sim.recommend()
        scenarios = df.drop(columns=["adjusted"]).reset_index().to_dict("records")
        simulated_outcomes = {
            "horizon_hours": self.horizon,
            "scenarios": scenarios,
            "twin_recommended_scenario": best,
        }

        action = {
            "maintenance_stop": "Schedule a maintenance stop now.",
            "reduced_load": "Keep running at reduced load and re-inspect next shift.",
            "continue": "Continue production under increased monitoring.",
        }[best]

        bits = [f"{action} Risk level {risk_level} "
                f"(score {risk_score}: failure probability {fp:.0%}, "
                f"RUL estimate {rul:.0f} h"
                + (f", visual defect severity {severity:.0%}" if defect else "")
                + ")."]
        if defect:
            bits.append("A surface defect was detected on the current part; "
                        "quarantine it and sample the last batch.")
        if knowledge_out.get("answer"):
            src = ""
            if knowledge_out.get("source_doc"):
                src = (f" [ref: {knowledge_out['source_doc']} section "
                       f"{knowledge_out.get('source_section')}]")
            bits.append("Procedure: " + knowledge_out["answer"].strip() + src)

        return {
            "recommendation": " ".join(bits),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "simulated_outcomes": simulated_outcomes,
            "requires_human_approval": True,
        }


if __name__ == "__main__":
    v = {"defect": True, "severity": 0.7, "confidence": 0.9}
    p = {"failure_prob": 0.62, "rul_estimate": 88, "confidence": 0.8}
    k = {"answer": "Reduce feed rate and bring axis load below 50%.",
         "source_doc": "safety_sop.pdf", "source_section": "3.0"}
    out = PlanningAgent().run(v, p, k)
    import json
    print(json.dumps(out, indent=2, default=str))

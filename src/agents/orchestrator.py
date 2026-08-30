"""Orchestrator - chain the four agents in sequence (plain-Python, no framework).

    image ----------> VisionAgent ------\
    sensor_window --> PredictiveAgent ---> PlanningAgent --> decision package
    query ----------> KnowledgeAgent ---/

Returns a single dict with every agent's structured output plus a combined
`decision` block. Nothing here acts on the plant - the result is decision
support for a human approver.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.knowledge_agent import KnowledgeAgent
from agents.planning_agent import PlanningAgent
from agents.predictive_agent import PredictiveAgent
from agents.vision_agent import VisionAgent

MODE_QUERY = {
    "OSF": "What should I do on an overstrain OSF alert?",
    "HDF": "What should I do on a heat dissipation HDF alert?",
    "PWF": "What should I do on a power PWF alert?",
    "TWF": "When must a worn tool be changed?",
    "RNF": "How should a random failure RNF flag be handled?",
}


class Orchestrator:
    def __init__(self, lazy=True):
        self._vision = None
        self._pred = None
        self._know = None
        self._plan = PlanningAgent()
        if not lazy:
            self.vision, self.predictive, self.knowledge

    # lazy singletons so the Streamlit app only pays for what it uses
    @property
    def vision(self):
        if self._vision is None:
            self._vision = VisionAgent()
        return self._vision

    @property
    def predictive(self):
        if self._pred is None:
            self._pred = PredictiveAgent()
        return self._pred

    @property
    def knowledge(self):
        if self._know is None:
            self._know = KnowledgeAgent()
        return self._know

    def run(self, image=None, sensor_window=None, query=None, failure_mode_hint=None):
        t0 = time.time()
        trace = []

        vision_out = {"defect": False, "severity": 0.0, "confidence": 0.0,
                      "skipped": True}
        if image is not None:
            vision_out = self.vision.run(image)
            trace.append("VisionAgent")

        predictive_out = {"failure_prob": 0.0, "rul_estimate": 125.0,
                          "confidence": 0.0, "skipped": True}
        if sensor_window is not None:
            predictive_out = self.predictive.run(sensor_window)
            trace.append("PredictiveAgent")

        if query is None and failure_mode_hint:
            query = MODE_QUERY.get(failure_mode_hint.upper(),
                                   "What is the recommended maintenance action?")
        context = ""
        if failure_mode_hint:
            context = f"failure mode {failure_mode_hint}"
        knowledge_out = {"answer": "", "source_doc": None, "source_section": None,
                         "skipped": True}
        if query:
            knowledge_out = self.knowledge.run(query, context=context)
            trace.append("KnowledgeAgent")

        decision = self._plan.run(vision_out, predictive_out, knowledge_out)
        trace.append("PlanningAgent")

        return {
            "vision": vision_out,
            "predictive": predictive_out,
            "knowledge": knowledge_out,
            "decision": decision,
            "agent_trace": trace,
            "elapsed_s": round(time.time() - t0, 2),
        }


if __name__ == "__main__":
    import glob
    import json

    import numpy as np
    import pandas as pd
    sys.path.insert(0, "src")
    import config as C

    # a real high-wear AI4I-style row
    sw = {"tabular": {"air_temp_K": 302.0, "process_temp_K": 311.5,
                      "rot_speed_rpm": 1330, "torque_Nm": 60, "tool_wear_min": 220,
                      "type": "L"}}
    # a real CMAPSS test window (engine 1, last cycles)
    cols = json.load(open(os.path.join(C.MODELS, "lstm_rul_config.json")))["sensor_cols"]
    te = pd.read_csv(C.CMAPSS_TEST, sep=r"\s+", header=None, names=C.CMAPSS_COLS)
    g = te[te.unit == 1].sort_values("cycle")
    sw["cmapss_window"] = g[cols].values[-50:].tolist()

    img = sorted(glob.glob(os.path.join(C.CASTING_DIR, "test", "def_front", "*.jpeg")))[0]

    out = Orchestrator().run(image=img, sensor_window=sw, failure_mode_hint="OSF")
    print(json.dumps(out, indent=2, default=str))

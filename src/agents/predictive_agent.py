"""PredictiveAgent - failure probability (tabular) + RUL estimate (LSTM).

run(sensor_window) -> {"failure_prob": float, "rul_estimate": float, "confidence": float}

`sensor_window` is a dict:
    {
      "tabular": {air_temp_K, process_temp_K, rot_speed_rpm, torque_Nm,
                  tool_wear_min, type: "L"|"M"|"H"},          # raw sensor values
      "cmapss_window": [[s1..s14], ...]   # optional: >=1 rows of the 14 kept
                                           # CMAPSS sensors, most recent last
    }
Either key may be omitted. The tabular preprocessor (imputer+scaler) and the
LSTM normalisation stats are loaded from models_registry so inference matches
training exactly.
"""
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

_PRE = os.path.join(C.MODELS, "tabular_preprocessor.joblib")
_CLF = os.path.join(C.MODELS, "tabular_best.joblib")
_LSTM = os.path.join(C.MODELS, "lstm_rul.keras")
_LSTM_CFG = os.path.join(C.MODELS, "lstm_rul_config.json")


def engineer_tabular(d):
    """Same features as src/data/prepare_tabular.py, from raw sensor values."""
    at, pt = d["air_temp_K"], d["process_temp_K"]
    rs, tq, tw = d["rot_speed_rpm"], d["torque_Nm"], d["tool_wear_min"]
    typ = str(d.get("type", "M")).upper()
    return {
        "air_temp_K": at, "process_temp_K": pt, "rot_speed_rpm": rs,
        "torque_Nm": tq, "tool_wear_min": tw,
        "temp_diff_K": pt - at,
        "power_W": tq * rs * 2 * np.pi / 60,
        "wear_torque": tw * tq,
        "type_L": int(typ == "L"), "type_M": int(typ == "M"),
        "type_H": int(typ == "H"),
    }


class PredictiveAgent:
    def __init__(self):
        self.pre = joblib.load(_PRE)
        self.clf_bundle = joblib.load(_CLF)
        self.clf = self.clf_bundle["model"]
        self.features = self.pre["features"]
        self._lstm = None
        self.lstm_cfg = json.load(open(_LSTM_CFG))

    def _lstm_model(self):
        if self._lstm is None:
            import tensorflow as tf
            self._lstm = tf.keras.models.load_model(_LSTM)
        return self._lstm

    def _failure_prob(self, tab):
        row = engineer_tabular(tab)
        X = np.array([[row[f] for f in self.features]], dtype=float)
        X = self.pre["scaler"].transform(self.pre["imputer"].transform(X))
        p = float(self.clf.predict_proba(X)[0, 1])
        return p

    def _rul(self, window):
        cfg = self.lstm_cfg
        cols = cfg["sensor_cols"]
        w = cfg["window"]
        arr = np.asarray(window, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != len(cols):
            raise ValueError(f"cmapss_window needs {len(cols)} sensors, "
                             f"got {arr.shape[1]}")
        mean = np.array([cfg["norm_mean"][c] for c in cols])
        std = np.array([cfg["norm_std"][c] for c in cols])
        arr = (arr - mean) / std
        if len(arr) < w:                       # left-pad with the oldest row
            arr = np.vstack([np.repeat(arr[:1], w - len(arr), axis=0), arr])
        arr = arr[-w:]
        pred = float(self._lstm_model().predict(arr[None], verbose=0).ravel()[0])
        return float(np.clip(pred, 0, cfg["rul_cap"]))

    def run(self, sensor_window):
        failure_prob, rul_estimate = None, None
        conf_parts = []

        if sensor_window.get("tabular"):
            failure_prob = self._failure_prob(sensor_window["tabular"])
            conf_parts.append(abs(failure_prob - 0.5) * 2)   # margin -> 0..1

        if sensor_window.get("cmapss_window") is not None:
            rul_estimate = self._rul(sensor_window["cmapss_window"])
            cap = self.lstm_cfg["rul_cap"]
            # closer to end-of-life -> more confident it's actionable
            conf_parts.append(float(np.clip(1 - rul_estimate / cap, 0.1, 1.0)))

        if failure_prob is None:
            failure_prob = 0.0
        if rul_estimate is None:
            rul_estimate = float(self.lstm_cfg["rul_cap"])   # unknown -> assume healthy
        confidence = float(np.mean(conf_parts)) if conf_parts else 0.0

        return {"failure_prob": round(failure_prob, 4),
                "rul_estimate": round(rul_estimate, 2),
                "confidence": round(confidence, 4)}


if __name__ == "__main__":
    pa = PredictiveAgent()
    demo = {"tabular": {"air_temp_K": 301.5, "process_temp_K": 311.2,
                        "rot_speed_rpm": 1320, "torque_Nm": 58, "tool_wear_min": 215,
                        "type": "L"}}
    print("high-stress row ->", pa.run(demo))
    demo2 = {"tabular": {"air_temp_K": 298, "process_temp_K": 308, "rot_speed_rpm": 1500,
                         "torque_Nm": 40, "tool_wear_min": 20, "type": "M"}}
    print("healthy row     ->", pa.run(demo2))

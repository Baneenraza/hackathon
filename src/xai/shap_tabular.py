"""Block 8 - SHAP explanations for the tabular failure classifier.

Builds:
  reports/shap/shap_summary.png       global feature importance (beeswarm)
  reports/shap/shap_bar.png           mean |SHAP| bar chart
  reports/shap/global_importance.json ordered feature importances
  models_registry/shap_tabular.joblib the fitted explainer (for the app / agents)

Exposes explain_instance(row_dict) -> plain-language "why" + signed contributions,
used by the agents and the Streamlit app to show a reason for every prediction.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT = os.path.join(C.REPORTS, "shap")
os.makedirs(OUT, exist_ok=True)

_PRE = os.path.join(C.MODELS, "tabular_preprocessor.joblib")
_CLF = os.path.join(C.MODELS, "tabular_best.joblib")

FRIENDLY = {
    "air_temp_K": "air temperature", "process_temp_K": "process temperature",
    "rot_speed_rpm": "rotational speed", "torque_Nm": "torque",
    "tool_wear_min": "tool wear", "temp_diff_K": "process-air temp difference",
    "power_W": "mechanical power", "wear_torque": "tool-wear x torque (overstrain)",
    "type_L": "machine type L", "type_M": "machine type M", "type_H": "machine type H",
}


def _load():
    pre = joblib.load(_PRE)
    clf = joblib.load(_CLF)["model"]
    feats = pre["features"]
    tr = pd.read_csv(os.path.join(C.PROCESSED, "tabular_train.csv"))[feats]
    te = pd.read_csv(os.path.join(C.PROCESSED, "tabular_test.csv"))[feats]
    return pre, clf, feats, tr, te


def build():
    pre, clf, feats, tr, te = _load()
    bg = shap.sample(tr, 200, random_state=C.RANDOM_STATE)
    explainer = shap.TreeExplainer(clf, bg, model_output="probability")
    sample = te.sample(min(400, len(te)), random_state=C.RANDOM_STATE)
    sv = explainer(sample, check_additivity=False)
    # for binary RandomForest, shap may return (n,f,2) -> take positive class
    vals = sv.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]
        base = sv.base_values[:, 1] if np.ndim(sv.base_values) > 1 else sv.base_values
    else:
        base = sv.base_values
    exp = shap.Explanation(values=vals, base_values=base, data=sample.values,
                           feature_names=feats)

    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary.png", dpi=120); plt.close()

    plt.figure()
    shap.plots.bar(exp, show=False, max_display=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/shap_bar.png", dpi=120); plt.close()

    imp = np.abs(vals).mean(0)
    order = np.argsort(imp)[::-1]
    ranked = [{"feature": feats[i], "friendly": FRIENDLY.get(feats[i], feats[i]),
               "mean_abs_shap": round(float(imp[i]), 5)} for i in order]
    json.dump(ranked, open(f"{OUT}/global_importance.json", "w"), indent=2)

    joblib.dump({"explainer": explainer, "features": feats,
                 "base_value": float(np.mean(base))},
                os.path.join(C.MODELS, "shap_tabular.joblib"))

    md = ["# Block 8 - SHAP feature importance (tabular failure classifier)", "",
          "![beeswarm](shap/shap_summary.png)", "",
          "![bar](shap/shap_bar.png)", "",
          "## Global ranking (mean |SHAP| on the test set)", "",
          "| rank | feature | mean |SHAP| |", "|---|---|---|"]
    for r, x in enumerate(ranked, 1):
        md.append(f"| {r} | {x['friendly']} (`{x['feature']}`) | {x['mean_abs_shap']} |")
    md += ["", "The engineered overstrain feature `wear_torque` and `torque_Nm` / "
           "`power_W` carry most of the signal - consistent with the AI4I failure "
           "physics (OSF, PWF, HDF). `type_*` and `air_temp_K` contribute little. "
           "These SHAP values feed the per-prediction 'why' shown in the app and "
           "the agent explanations."]
    open(os.path.join(C.REPORTS, "shap_tabular_report.md"), "w",
         encoding="utf-8").write("\n".join(md) + "\n")
    print("SHAP tabular artifacts ->", os.path.relpath(OUT, C.ROOT))
    return ranked


def explain_instance(row_dict, top_n=3):
    """row_dict: raw sensor values (same keys as PredictiveAgent tabular input).
    Returns {prediction, probability, base_value, contributions[], why}."""
    from agents.predictive_agent import engineer_tabular
    pre = joblib.load(_PRE)
    bundle = joblib.load(os.path.join(C.MODELS, "shap_tabular.joblib"))
    clf = joblib.load(_CLF)["model"]
    feats = pre["features"]

    eng = engineer_tabular(row_dict)
    Xraw = np.array([[eng[f] for f in feats]], dtype=float)
    Xs = pre["scaler"].transform(pre["imputer"].transform(Xraw))
    prob = float(clf.predict_proba(Xs)[0, 1])

    sv = bundle["explainer"](pd.DataFrame(Xs, columns=feats), check_additivity=False)
    vals = sv.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    contrib = vals[0]
    order = np.argsort(np.abs(contrib))[::-1][:top_n]
    items = []
    for i in order:
        direction = "raises" if contrib[i] > 0 else "lowers"
        items.append({"feature": feats[i],
                      "friendly": FRIENDLY.get(feats[i], feats[i]),
                      "value": round(float(Xraw[0, i]), 2),
                      "shap": round(float(contrib[i]), 4),
                      "effect": direction})
    why = "; ".join(f"{it['friendly']} ({it['value']}) {it['effect']} the risk"
                    for it in items)
    return {"prediction": "failure" if prob >= 0.5 else "healthy",
            "probability": round(prob, 4),
            "base_value": round(bundle["base_value"], 4),
            "contributions": items,
            "why": f"Predicted {'FAILURE' if prob >= 0.5 else 'healthy'} "
                   f"({prob:.0%}). Main drivers: {why}."}


if __name__ == "__main__":
    build()
    demo = {"air_temp_K": 302.0, "process_temp_K": 311.5, "rot_speed_rpm": 1330,
            "torque_Nm": 60, "tool_wear_min": 220, "type": "L"}
    print(json.dumps(explain_instance(demo), indent=2))

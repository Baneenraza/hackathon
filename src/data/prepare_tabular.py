"""Block 2 - AI4I 2020 tabular pipeline.

Steps (all logged to reports/tabular_prep_report.md):
  1. load raw
  2. INJECT realistic data-quality problems (duplicates + missing values) with a
     fixed seed, so we can demonstrate detection + handling (rubric requirement)
  3. detect & handle duplicates (exact-row drop, done BEFORE the split so no row
     leaks across train/test)
  4. feature engineering (domain features for the AI4I failure modes)
  5. stratified train / val / test split (60/20/20) on the binary target
  6. fit median imputer + standard scaler on TRAIN ONLY, transform val/test
  7. persist cleaned splits + fitted transformers + feature list

Leakage control:
  - TWF/HDF/PWF/OSF/RNF are the *components* of `Machine failure`; they are removed
    from the feature matrix and kept aside only for error analysis in Block 3.
  - UDI / Product ID are identifiers -> dropped.
  - imputer and scaler are never shown validation/test statistics.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT = C.PROCESSED
REPORT = os.path.join(C.REPORTS, "tabular_prep_report.md")
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def inject_quality_issues(df, seed=C.RANDOM_STATE):
    """Add duplicates and missing values so handling can be demonstrated."""
    rng = np.random.default_rng(seed)
    n = len(df)

    # --- duplicates: copy 180 random rows verbatim (~1.8%) ---
    dup_idx = rng.choice(n, size=180, replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # --- missing values: blank out cells in 4 sensor columns (~2.5% each) ---
    miss_cols = ["Air temperature [K]", "Process temperature [K]",
                 "Rotational speed [rpm]", "Tool wear [min]"]
    for col in miss_cols:
        m = rng.choice(len(df), size=int(0.025 * len(df)), replace=False)
        df.loc[m, col] = np.nan

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, len(dup_idx), miss_cols


def engineer_features(df):
    """Domain features tied to the AI4I failure physics."""
    out = df.copy()
    out["temp_diff_K"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    # mechanical power = torque * angular velocity (rpm -> rad/s)
    out["power_W"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * 2 * np.pi / 60
    # overstrain driver (OSF): tool wear * torque
    out["wear_torque"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    # one-hot machine type (L / M / H)
    for t in ["L", "M", "H"]:
        out[f"type_{t}"] = (out["Type"] == t).astype(int)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log("# Block 2 - AI4I Tabular Preparation Report\n")

    raw = pd.read_csv(C.TABULAR_CSV)
    log(f"Raw shape: {raw.shape}")
    log(f"Class balance (raw): {raw[TARGET].value_counts().to_dict()} "
        f"({raw[TARGET].mean()*100:.2f}% positive)\n")

    # ---------------- 1. inject issues ----------------
    dirty, n_dup, miss_cols = inject_quality_issues(raw)
    log("## 1. Injected data-quality issues (seeded, reproducible)")
    log(f"- Duplicated {n_dup} random rows verbatim -> shape now {dirty.shape}")
    log(f"- Inserted NaNs into {miss_cols} at ~2.5% each")
    log(f"- Missing-value counts after injection:\n```\n"
        f"{dirty[miss_cols].isna().sum().to_string()}\n```\n")

    # ---------------- 2. handle duplicates ----------------
    log("## 2. Duplicate handling")
    n_exact = dirty.duplicated().sum()
    log(f"- Exact duplicate rows detected: {n_exact}")
    dirty = dirty.drop_duplicates().reset_index(drop=True)
    # also check identifier-level dups (same UDI) that slipped past due to NaNs
    id_dups = dirty.duplicated(subset=["UDI"]).sum()
    if id_dups:
        dirty = dirty.drop_duplicates(subset=["UDI"]).reset_index(drop=True)
    log(f"- Additional UDI-level duplicates removed: {id_dups}")
    log(f"- Shape after de-duplication: {dirty.shape}\n")

    # ---------------- 3. drop leakage / id cols, engineer ----------------
    log("## 3. Leakage control & feature engineering")
    log(f"- Dropped identifiers: UDI, Product ID")
    log(f"- Held out (NOT features, used for Block 3 error analysis): {FAILURE_MODES}")
    y = dirty[TARGET].astype(int)
    failure_modes_df = dirty[FAILURE_MODES].astype(int)
    feat = engineer_features(dirty)
    engineered = ["temp_diff_K", "power_W", "wear_torque", "type_L", "type_M", "type_H"]
    log(f"- Engineered features: {engineered}")

    # XGBoost rejects '[', ']', '<' in feature names -> use safe slugs everywhere
    rename = {"Air temperature [K]": "air_temp_K",
              "Process temperature [K]": "process_temp_K",
              "Rotational speed [rpm]": "rot_speed_rpm",
              "Torque [Nm]": "torque_Nm",
              "Tool wear [min]": "tool_wear_min"}
    feat = feat.rename(columns=rename)
    numeric_base = list(rename.values())
    feature_cols = numeric_base + engineered
    X = feat[feature_cols].copy()
    log(f"- Final feature matrix: {X.shape[1]} features\n")

    # ---------------- 4. stratified split ----------------
    log("## 4. Stratified train / val / test split (60 / 20 / 20)")
    idx = np.arange(len(X))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.4, stratify=y, random_state=C.RANDOM_STATE)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, stratify=y.iloc[idx_tmp], random_state=C.RANDOM_STATE)

    def take(a):
        return X.iloc[a].reset_index(drop=True), y.iloc[a].reset_index(drop=True)

    X_tr, y_tr = take(idx_train)
    X_va, y_va = take(idx_val)
    X_te, y_te = take(idx_test)
    for name, yy in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        log(f"- {name}: {len(yy)} rows, {int(yy.sum())} positives "
            f"({yy.mean()*100:.2f}%)")
    log("")

    # ---------------- 5. impute + scale (fit on train only) ----------------
    log("## 5. Missing-value imputation + scaling (fit on TRAIN only -> no leakage)")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_i = imputer.fit_transform(X_tr)
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_va_s = scaler.transform(imputer.transform(X_va))
    X_te_s = scaler.transform(imputer.transform(X_te))
    med = dict(zip(feature_cols, np.round(imputer.statistics_, 3)))
    log(f"- Train medians used for imputation: {med}")
    log(f"- NaNs remaining after impute: train={np.isnan(X_tr_i).sum()}\n")

    # ---------------- 6. persist ----------------
    def save_split(name, Xs, yy, rows_idx):
        d = pd.DataFrame(Xs, columns=feature_cols)
        d[TARGET] = yy.values
        fm = failure_modes_df.iloc[rows_idx].reset_index(drop=True)
        for c in FAILURE_MODES:
            d[c] = fm[c].values
        p = os.path.join(OUT, f"tabular_{name}.csv")
        d.to_csv(p, index=False)
        return p

    p1 = save_split("train", X_tr_s, y_tr, idx_train)
    p2 = save_split("val", X_va_s, y_va, idx_val)
    p3 = save_split("test", X_te_s, y_te, idx_test)
    joblib.dump({"imputer": imputer, "scaler": scaler, "features": feature_cols},
                os.path.join(C.MODELS, "tabular_preprocessor.joblib"))
    # also unscaled cleaned full frame for EDA / SHAP readability
    feat_clean = feat[feature_cols].copy()
    feat_clean[TARGET] = y.values
    feat_clean.to_csv(os.path.join(OUT, "tabular_clean_unscaled.csv"), index=False)

    log("## 6. Artifacts written")
    for p in [p1, p2, p3,
              os.path.join(C.MODELS, "tabular_preprocessor.joblib"),
              os.path.join(OUT, "tabular_clean_unscaled.csv")]:
        log(f"- {os.path.relpath(p, C.ROOT)}")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")
    log(f"\nReport -> {os.path.relpath(REPORT, C.ROOT)}")
    log("BLOCK 2 OK")


if __name__ == "__main__":
    main()

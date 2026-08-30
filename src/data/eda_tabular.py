"""Block 2 - quick EDA visuals for the AI4I tabular data -> reports/eda/."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT = os.path.join(C.REPORTS, "eda")
os.makedirs(OUT, exist_ok=True)


def main():
    raw = pd.read_csv(C.TABULAR_CSV)
    clean = pd.read_csv(os.path.join(C.PROCESSED, "tabular_clean_unscaled.csv"))
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    # 1. class balance + failure-mode breakdown
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    raw["Machine failure"].value_counts().plot.bar(
        ax=ax[0], color=["#4c72b0", "#c44e52"])
    ax[0].set_title("Target balance (0=OK, 1=failure)")
    ax[0].set_xticklabels(["OK", "failure"], rotation=0)
    raw[modes].sum().plot.bar(ax=ax[1], color="#c44e52")
    ax[1].set_title("Failure count by mode")
    fig.tight_layout(); fig.savefig(f"{OUT}/01_class_balance.png", dpi=110); plt.close(fig)

    # 2. feature distributions split by outcome
    feats = ["Air temperature [K]", "Process temperature [K]",
             "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for f, a in zip(feats, axes.ravel()):
        for lab, col in [(0, "#4c72b0"), (1, "#c44e52")]:
            a.hist(raw.loc[raw["Machine failure"] == lab, f], bins=40, alpha=0.6,
                   density=True, color=col, label=f"fail={lab}")
        a.set_title(f); a.legend()
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/02_feature_dists.png", dpi=110); plt.close(fig)

    # 3. correlation heatmap (engineered clean set)
    corr = clean.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Feature correlation (engineered, cleaned)")
    fig.tight_layout(); fig.savefig(f"{OUT}/03_correlation.png", dpi=110); plt.close(fig)

    # 4. failure rate by machine type
    fig, ax = plt.subplots(figsize=(6, 4))
    (raw.groupby("Type")["Machine failure"].mean() * 100).plot.bar(
        ax=ax, color="#dd8452")
    ax.set_ylabel("failure rate %"); ax.set_title("Failure rate by machine Type")
    fig.tight_layout(); fig.savefig(f"{OUT}/04_failure_by_type.png", dpi=110); plt.close(fig)

    # text summary
    lines = ["# AI4I EDA summary", "",
             f"- rows={len(raw)}, positive rate={raw['Machine failure'].mean()*100:.2f}%",
             f"- failure modes (raw counts): {raw[modes].sum().to_dict()}",
             f"- overlap (rows with >1 mode): "
             f"{int((raw[modes].sum(axis=1) > 1).sum())}",
             f"- failure rate by type: "
             f"{(raw.groupby('Type')['Machine failure'].mean()*100).round(2).to_dict()}",
             "- strongest |corr| with target: "
             + ", ".join(
                 f"{k}={v:.2f}" for k, v in
                 corr['Machine failure'].drop('Machine failure').abs()
                 .sort_values(ascending=False).head(5).items()),
             "", "Plots: 01_class_balance, 02_feature_dists, 03_correlation, "
             "04_failure_by_type (PNG in this folder)."]
    with open(f"{OUT}/eda_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nEDA plots ->", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()

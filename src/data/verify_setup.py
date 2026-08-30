"""Block 1 smoke test: confirm all 3 downloaded datasets + generated assets load."""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def main():
    checks = []

    ai = pd.read_csv(C.TABULAR_CSV)
    checks.append(("AI4I tabular", ai.shape == (10000, 14),
                   f"{ai.shape}, {int(ai['Machine failure'].sum())} failures"))

    tr = pd.read_csv(C.CMAPSS_TRAIN, sep=r"\s+", header=None, names=C.CMAPSS_COLS)
    te = pd.read_csv(C.CMAPSS_TEST, sep=r"\s+", header=None, names=C.CMAPSS_COLS)
    rul = pd.read_csv(C.CMAPSS_RUL, sep=r"\s+", header=None)
    checks.append(("CMAPSS FD001", tr.shape == (20631, 26) and len(rul) == 100,
                   f"train {tr.shape} ({tr.unit.nunique()} units), "
                   f"test {te.shape}, RUL {rul.shape}"))

    img = {f"{s}/{c}": len(glob.glob(f"{C.CASTING_DIR}/{s}/{c}/*.jpeg"))
           for s in ("train", "test") for c in ("def_front", "ok_front")}
    checks.append(("casting images", sum(img.values()) > 7000, str(img)))

    mn = pd.read_csv(C.NOTES_CSV)
    checks.append(("maintenance_notes", mn.shape == (450, 8),
                   f"{mn.shape}, cats={sorted(mn.category.unique())}, "
                   f"urgency={sorted(mn.urgency.unique())}, "
                   f"null hints={int(mn.failure_type_hint.isna().sum())}"))

    kb = sorted(os.path.basename(p) for p in glob.glob(f"{C.KNOWLEDGE_BASE}/*.pdf"))
    checks.append(("knowledge base PDFs", len(kb) == 3, str(kb)))

    checks.append(("GROQ_API_KEY", bool(C.load_groq_key()), "loaded from .env"))

    ok = True
    for name, passed, detail in checks:
        flag = "PASS" if passed else "FAIL"
        ok &= passed
        print(f"[{flag}] {name:22s} {detail}")
    print("\nBLOCK 1 SETUP OK" if ok else "\nBLOCK 1 SETUP INCOMPLETE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

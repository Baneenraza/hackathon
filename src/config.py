"""Central paths and settings for the AI Factory Intelligence Command Center."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
PROCESSED = os.path.join(DATA, "processed")
KNOWLEDGE_BASE = os.path.join(DATA, "knowledge_base")

TABULAR_CSV = os.path.join(RAW, "tabular", "ai4i2020.csv")
CMAPSS_TRAIN = os.path.join(RAW, "timeseries", "train_FD001.txt")
CMAPSS_TEST = os.path.join(RAW, "timeseries", "test_FD001.txt")
CMAPSS_RUL = os.path.join(RAW, "timeseries", "RUL_FD001.txt")
CASTING_DIR = os.path.join(RAW, "images", "casting")
NOTES_CSV = os.path.join(RAW, "notes", "maintenance_notes.csv")

MODELS = os.path.join(ROOT, "models_registry")
MLRUNS = os.path.join(ROOT, "mlruns")
REPORTS = os.path.join(ROOT, "reports")
for _d in (PROCESSED, MODELS, REPORTS):
    os.makedirs(_d, exist_ok=True)

# MLflow's file store is now maintenance-mode/blocked -> use a local sqlite backend.
MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(ROOT, "mlflow.db").replace("\\", "/")
MLFLOW_ARTIFACT_URI = MLRUNS.replace("\\", "/")

# CMAPSS column names (FD001): unit, cycle, 3 op settings, 21 sensors
CMAPSS_COLS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

RANDOM_STATE = 42

# --- LLM (supporting layer only, never the primary predictor) ---
# NOTE: this Groq key has no access to the llama-3.1-8b-instant model (404).
# openai/gpt-oss-20b is the small, fast, instruction-tuned model available here.
GROQ_MODEL = "openai/gpt-oss-20b"


def load_groq_key():
    """Order: process env  ->  .env file  ->  Streamlit secrets (cloud deploy)."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(ROOT, ".env"))
            key = os.getenv("GROQ_API_KEY")
        except Exception:
            pass
    if not key:
        try:
            import streamlit as st
            try:
                key = st.secrets["GROQ_API_KEY"]
            except Exception:
                key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            pass
    if key:
        key = str(key).strip().strip('"').strip("'")
    return key or None

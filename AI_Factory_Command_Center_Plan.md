# AI Factory Intelligence Command Center — Hackathon Execution Plan
Baneen Raza — Sylani Data Science with AI Hackathon — Deadline: today 4:00 PM

---

## 0. Scope Decision (read first)

This rubric spans 9 stages worth ~90+ marks total in one day. You will NOT go deep everywhere.
**Rule: every rubric bullet must be touched at MVP level. Nothing skipped, nothing gold-plated.**

Narrative used to justify tying 5 different real/synthetic datasets into "one factory":
> "The factory runs a fleet of CNC/milling machines (AI4I) and a fleet of rotating equipment
> monitored continuously (CMAPSS-style sensors), produces cast components inspected visually
> (casting image set), and technicians log incidents in text tied to machine IDs, with SOPs/manuals
> as PDF knowledge base." State this explicitly as a documented assumption in the report — the
> rubric explicitly asks you to document assumptions, so this is a strength, not a weakness.

---

## 1. Tech Stack (final — do not re-litigate mid-hackathon)

| Layer | Choice | Why |
|---|---|---|
| Data/EDA | Pandas, NumPy, Matplotlib/Plotly | fastest, everyone knows it |
| Baseline ML | Scikit-learn RandomForest + XGBoost | 10 min to train, strong baseline |
| Deep Learning (time-series) | Keras LSTM | 1 architecture only — RUL regression on CMAPSS |
| Deep Learning (vision) | Keras + MobileNetV2 transfer learning | fast convergence, small dataset |
| NLP | TF-IDF + LogisticRegression (urgency/category) + simple regex/keyword extraction (machine ID, part) | no transformer fine-tuning — no time |
| RAG embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | free, no API latency |
| Vector store | FAISS (local) | zero infra |
| LLM (generation/explanation) | Groq API (`llama-3.1-8b-instant`) — free tier, fast. Fallback: your OpenAI/Anthropic key if you have one | must be a "supporting" layer only per rubric — never the predictor |
| Agents | Plain Python classes, structured dict I/O between them | fully satisfies "3+ agents, structured info passing" without framework overhead |
| Explainability | SHAP (tabular/LSTM), Grad-CAM (CNN) | standard, library-supported |
| MLOps | MLflow (local `mlruns/` file store) | log params/metrics/artifacts, register best model |
| Digital Twin | Custom Python `FactorySimulator` class | simplest way to hit "3 scenarios + compare outcomes" |
| Web App | Streamlit | fastest full working UI |
| Report | `fpdf2` | generates PDF in-app, no template complexity |

---

## 2. Datasets (verified working links, Aug 2026)

| Modality | Dataset | Link | Notes |
|---|---|---|---|
| Tabular | AI4I 2020 Predictive Maintenance | kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020 | 10k rows, 5 failure types, no missing values (inject some synthetically for the "handle missing values" rubric point) |
| Time-series | NASA CMAPSS Turbofan (FD001) | kaggle.com/datasets/behrad3d/nasa-cmaps | 100 engines, run-to-failure, 21 sensors — use FD001 only, skip FD002-004 |
| Image | Casting Product QA | kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product | binary ok/defective, ~7300 images, already split train/test |
| Text | Synthetic maintenance notes | generate yourself (see §7 prompt) | ~150-200 short notes tagged machine_id, urgency, category |
| PDF | Synthetic SOPs/manuals | generate yourself, 3-4 short PDFs (2-3 pages each) | machine safety SOP, maintenance manual excerpt, calibration procedure |

Download via `kagglehub` or `kaggle datasets download -d <slug>` (needs kaggle.json API token — set this up FIRST, it's the #1 time-waster).

---

## 3. Repo Structure (hand this to Claude Code verbatim)

```
factory-ai/
├── data/
│   ├── raw/                    # downloaded datasets, untouched
│   ├── processed/              # cleaned/feature-engineered
│   └── knowledge_base/         # synthetic PDFs + notes for RAG
├── notebooks/
│   ├── 01_eda_tabular.ipynb
│   ├── 02_eda_timeseries.ipynb
│   ├── 03_baseline_ml.ipynb
│   ├── 04_deep_learning.ipynb
│   ├── 05_cv_defect.ipynb
│   └── 06_nlp_notes.ipynb
├── src/
│   ├── data/                   # loaders, cleaning, feature engineering
│   ├── models/                 # train/predict scripts per modality
│   ├── xai/                    # shap_utils.py, gradcam_utils.py
│   ├── rag/                    # embed.py, index.py, retrieve.py
│   ├── agents/                 # vision_agent.py, predictive_agent.py, knowledge_agent.py, planning_agent.py, orchestrator.py
│   ├── digital_twin/           # simulator.py
│   └── report/                 # pdf_generator.py
├── mlruns/                     # mlflow tracking (auto-created)
├── app/
│   └── streamlit_app.py
├── models_registry/            # saved best models (.pkl, .h5)
├── README.md                   # data sources, assumptions, limitations, architecture
└── requirements.txt
```

---

## 4. Build Order — Time-Boxed (compress proportionally to hours actually left)

| Block | Duration | Tasks |
|---|---|---|
| 1 | 45 min | Kaggle API setup, download all 3 real datasets, scaffold repo, `git init`, requirements.txt |
| 2 | 60 min | Tabular EDA + cleaning (inject nulls/dupes to legitimately demo handling), feature engineering, train/val/test split (time-aware for CMAPSS — no leakage) |
| 3 | 60 min | Baseline RF/XGBoost on tabular (failure classification) + LSTM on CMAPSS (RUL regression). Log both to MLflow. Compare metrics. |
| 4 | 45 min | CNN transfer learning on casting images (defect classification). Grad-CAM on a few samples. |
| 5 | 30 min | Generate synthetic maintenance notes + 3 SOP PDFs (use LLM to draft, see §7). TF-IDF classifier for urgency/category. |
| 6 | 60 min | Build RAG: chunk PDFs, embed, FAISS index, retrieval + LLM answer generation. Test 2-3 QA pairs, show retrieval beats no-retrieval on one example. |
| 7 | 60 min | Build 4 agents + orchestrator (Vision → Predictive → Knowledge → Planning, structured dict handoff). |
| 8 | 30 min | SHAP for tabular/LSTM feature importance. Wire XAI outputs into agent output (confidence + reason). |
| 9 | 45 min | Digital twin simulator: 3 scenarios (continue / stop-for-maintenance / reduce load), cost/downtime/risk comparison table. |
| 10 | 30 min | MLflow: register best model per modality, log 3+ experiment iterations (e.g. RF vs XGB vs tuned XGB). |
| 11 | 75 min | Streamlit app: input controls per modality, prediction + confidence + explanation display, digital twin what-if panel, APPROVE/REJECT/MODIFY buttons (log decision to a CSV/DB), PDF report download. |
| 12 | 30 min | README: data sources, assumptions, limitations, architecture diagram, "decision support not autonomous authority" disclaimer. |
| Buffer | remaining | Bug fixes, polish, rehearse demo narrative (walk through the 4 questions in §Challenge Overview: what's happening / what's next / why / what action). |

**If time runs out, cut in this order:** GRU/Transformer comparisons (keep only LSTM) → reduce MLflow experiments to exactly 3 → simplify digital twin cost model to basic arithmetic → skip PDF report styling (plain text PDF is fine) → reduce agents' sophistication but never drop below 3 agents.

---

## 5. Agent Design (exact contract for Claude Code)

```python
# Each agent returns a structured dict — this IS the "structured information passing" the rubric wants
VisionAgent.run(image) -> {"defect": bool, "severity": float, "confidence": float}
PredictiveAgent.run(sensor_window) -> {"failure_prob": float, "rul_estimate": float, "confidence": float}
KnowledgeAgent.run(query, context_dict) -> {"answer": str, "source_doc": str, "source_section": str}
PlanningAgent.run(vision_out, predictive_out, knowledge_out) -> {
    "recommendation": str, "risk_level": str, "simulated_outcomes": dict, "requires_human_approval": True
}
Orchestrator.run(inputs) -> calls all four in sequence, passes outputs forward, returns final PlanningAgent payload
```

---

## 6. Digital Twin — Minimal Spec

```python
class FactorySimulator:
    def __init__(self, base_production_rate, current_downtime_risk, maintenance_cost_per_hr, lost_production_value_per_unit):
        ...
    def simulate_continue(self, horizon_hours) -> {"expected_units": int, "downtime_hours": float, "cost": float, "risk_score": float}
    def simulate_maintenance_stop(self, horizon_hours, stop_duration) -> {...same keys...}
    def simulate_reduced_load(self, horizon_hours, load_factor) -> {...same keys...}
```
Feed `failure_prob` / `rul_estimate` from PredictiveAgent into the risk_score calc so the twin is actually driven by the model, not hardcoded — rubric explicitly penalizes hardcoded demo outputs.

---

## 7. Claude Code Handoff Prompt — paste this directly into Claude Code

```
I'm building "AI Factory Intelligence Command Center" for a graded hackathon, deadline today.
Build in the repo structure and stack below. Work fast, MVP-first, every component must actually
run end-to-end — no hardcoded fake outputs anywhere (this is explicitly penalized in grading).

STACK: Python, Pandas/Scikit-learn/XGBoost, TensorFlow/Keras (LSTM + MobileNetV2 transfer learning),
TF-IDF+LogisticRegression for NLP, sentence-transformers + FAISS for RAG, Groq API
(llama-3.1-8b-instant) for LLM generation, SHAP + Grad-CAM for XAI, MLflow for tracking,
plain-Python multi-agent orchestration (no LangGraph), Streamlit for the app, fpdf2 for reports.

DATASETS (download via kaggle API, I'll provide kaggle.json):
- kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020 (tabular)
- kaggle.com/datasets/behrad3d/nasa-cmaps, use FD001 only (time-series)
- kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product (images)

For text + PDF knowledge base, GENERATE synthetic data:
- ~150-200 short maintenance notes, each tagged with machine_id (link to AI4I machine types),
  urgency (low/medium/high), category (mechanical/electrical/software/safety), free-text description
- 3-4 short PDF documents (2-3 pages): a machine safety SOP, a maintenance procedure manual,
  a calibration checklist — realistic industrial tone, each with clear sections so RAG retrieval
  can cite "source document + section"

BUILD ORDER (see full plan doc I'm attaching for detail on every stage):
1. Repo scaffold + data download
2. Tabular EDA/cleaning/feature engineering (inject some nulls/duplicates first, then handle them —
   document this clearly, it's a rubric requirement) with proper train/val/test split, no leakage
3. Baseline RF + XGBoost (failure classification) vs LSTM (RUL regression on CMAPSS) — log both to
   MLflow, compare metrics (Precision/Recall/F1/ROC-AUC for classification, MAE/RMSE for regression),
   do basic error analysis (which failure types are misclassified and why)
4. CNN transfer learning (MobileNetV2) for casting defect classification + Grad-CAM visualization
5. TF-IDF + LogisticRegression on the synthetic maintenance notes for urgency/category classification
   + simple regex-based machine ID / part extraction
6. RAG: chunk the synthetic PDFs, embed with sentence-transformers, FAISS index, retrieval-augmented
   generation via Groq. Must cite source doc/section. Include one clear example comparing an answer
   WITH retrieval vs WITHOUT (unsupported generation) to demonstrate RAG's value.
7. Four agents with structured dict I/O exactly as specified in section 5 of the plan doc:
   VisionAgent, PredictiveAgent, KnowledgeAgent, PlanningAgent + an Orchestrator that chains them
8. SHAP for tabular/LSTM feature importance, feed into agent explanations — show confidence score
   and plain-language "why" for every prediction
9. FactorySimulator digital twin (see section 6 of plan doc) — 3 scenarios, driven by real model
   outputs (failure_prob/rul_estimate), not hardcoded numbers. Output a comparison table of
   production loss / downtime / risk / cost across scenarios.
10. MLflow: register the best model per modality, demonstrate at least 3 logged experiment
    iterations (e.g. RF baseline -> XGBoost -> tuned XGBoost)
11. Streamlit app with: file/data upload per modality, prediction + confidence + explanation
    display (show SHAP/Grad-CAM visuals), digital twin what-if panel, and REQUIRED human-in-the-loop
    controls — APPROVE / REJECT / MODIFY buttons that log the human decision + optional reason to a
    local CSV, and only after a decision is logged, generate a downloadable PDF incident/decision
    report via fpdf2 summarizing prediction, explanation, simulation results, and the human decision
12. README.md documenting: data sources, all assumptions (especially the synthetic text/PDF
    decision and the "unified factory" narrative tying datasets together), preprocessing decisions,
    limitations, system architecture diagram, and an explicit line stating the AI produces decision
    support only, never autonomous authority over real operations

Confirm the repo scaffold and Block 1 setup first, then proceed block by block per the time-boxed
plan — flag me immediately if any block is taking longer than budgeted so we can cut scope per the
priority-cut list in section 4 of the plan doc, rather than losing coverage on later stages.
```

---

## 8. Grading Safety Checklist (do this in the last 15 min, before 4 PM)

- [ ] Every rubric section (1-9 in original doc) has visible evidence in the app or notebooks
- [ ] README documents data sources, assumptions, limitations — explicitly
- [ ] At least one clear "RAG beats no-retrieval" example is shown
- [ ] At least 3 agents, structured data passing shown/visible (print or log the dict handoffs)
- [ ] Prediction confidence + human-readable explanation shown for every model output
- [ ] Digital twin shows 3 scenarios with a comparison, driven by actual model output
- [ ] APPROVE/REJECT/MODIFY controls exist and log a decision — AI output is never auto-final
- [ ] MLflow has 3+ logged runs and a registered best model
- [ ] Downloadable PDF/DOCX report generates successfully
- [ ] No hardcoded/fake demo outputs anywhere — if asked, you can explain how any number was produced

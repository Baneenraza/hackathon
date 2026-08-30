# AI Factory Intelligence Command Center

A single decision-support console that fuses four ML modalities over one
industrial narrative, explains every prediction, simulates the consequences of
each maintenance choice, and routes the final call to a named human.

> **The system produces decision support only. It has no autonomous authority.**
> Every prediction is probabilistic and advisory; a named human must APPROVE,
> REJECT or MODIFY each recommendation and that decision is logged.

---

## 1. The unified-factory narrative

The three public datasets are treated as three stations on one plant:

| Station | Dataset | Model | Question answered |
|---|---|---|---|
| CNC machining line | **AI4I 2020** (10 k rows, tabular) | RandomForest / XGBoost | Will this machine fail now, and from which mode (TWF/HDF/PWF/OSF/RNF)? |
| Rotating-equipment fleet | **NASA CMAPSS FD001** (100 engines, time-series) | Keras LSTM | How many hours of useful life remain (RUL)? |
| Casting QA station | **Casting product images** (7.3 k images) | MobileNetV2 transfer learning | Is the part that just came off the line defective? |
| Maintenance back-office | `maintenance_notes.csv` + 3 SOP/manual PDFs | TF-IDF + LogReg, RAG | Classify the note; what does the procedure say to do? |

The synthetic maintenance notes are deliberately aligned to the AI4I schema
(`machine_type` L/M/H, `failure_type_hint` TWF/HDF/PWF/OSF/RNF) so text, tabular
and knowledge-base evidence describe the *same* machines.

---

## 2. Architecture

```
                       ┌────────────────────────────────────────────┐
  casting image ─────► │ VisionAgent      MobileNetV2 + Grad-CAM     │─┐
                       ├────────────────────────────────────────────┤ │
  process sensors ───► │ PredictiveAgent  RF/XGB  (failure prob)     │ │
  turbofan history ──► │                  LSTM    (RUL) + SHAP       │ ├─► PlanningAgent ─► decision package
                       ├────────────────────────────────────────────┤ │      · risk level (rule-based fusion)
  failure-mode hint ─► │ KnowledgeAgent   SBERT + FAISS + Groq LLM   │ │      · FactorySimulator: 3 what-if scenarios
      / free query     │                  (cites doc + section)      │─┘      · recommendation (+ SOP citation)
                       └────────────────────────────────────────────┘        · requires_human_approval = True
                                                                                          │
                     Streamlit UI  ◄──────────────────────────────────────────────────────┘
                       prediction + confidence + SHAP/Grad-CAM + twin table
                       │
                       ▼
             Human: APPROVE / REJECT / MODIFY  + reason  ──►  decisions_log.csv
                       │
                       ▼
             fpdf2 incident & decision report (.pdf)
```

Orchestration is plain Python (`src/agents/orchestrator.py`) — the four agents
run in sequence, each with a strict `dict` I/O contract:

```python
VisionAgent.run(image)            -> {"defect": bool, "severity": float, "confidence": float}
PredictiveAgent.run(sensor_window)-> {"failure_prob": float, "rul_estimate": float, "confidence": float}
KnowledgeAgent.run(query, context)-> {"answer": str, "source_doc": str, "source_section": str}
PlanningAgent.run(v_out, p_out, k_out)
    -> {"recommendation": str, "risk_level": str,
        "simulated_outcomes": dict, "requires_human_approval": True}
```

---

## 3. Data sources

| Dataset | Origin | Local path |
|---|---|---|
| AI4I 2020 Predictive Maintenance | UCI / Kaggle | `data/raw/tabular/ai4i2020.csv` |
| NASA CMAPSS Turbofan Degradation (FD001) | NASA PCoE | `data/raw/timeseries/{train,test,RUL}_FD001.txt` |
| Casting product image data (def_front / ok_front) | Kaggle | `data/raw/images/casting/{train,test}/…` |
| Maintenance notes (450, synthetic) | generated for this project | `data/raw/notes/maintenance_notes.csv` |
| Safety SOP / Maintenance manual / Calibration checklist | **generated** (`src/data/make_knowledge_base.py`) | `data/knowledge_base/*.pdf` |

Run `python src/data/verify_setup.py` to confirm every source loads.

---

## 4. Assumptions & decisions

- **Synthetic text / PDF.** The maintenance notes and the three knowledge-base
  PDFs are synthetic. The notes were supplied; the PDFs are generated with
  numbered sections whose thresholds (200/240 min tool life, 8.6 K temperature
  delta, Type L/M/H overstrain limits) are internally consistent with the AI4I
  failure physics so cross-modality reasoning is meaningful. This is a
  deliberate scope decision, not hidden.
- **Unified-factory framing** (section 1) is an analytical convenience — the
  three real datasets are unrelated in origin.
- **LLM role.** The Groq LLM (`openai/gpt-oss-20b`; the planned
  `llama-3.1-8b-instant` is no longer served by Groq) is a *phrasing* layer
  only. It never predicts failure, RUL or defects, and the RAG pipeline refuses
  rather than answering when retrieval finds nothing above threshold.
- **Small NLP corpus / overfitting risk — documented, not hidden.** With 450
  templated notes the `category` classifier reaches ~1.0 five-fold CV accuracy;
  this is flagged in `reports/nlp_report.md` as *trivial separability from
  give-away keywords*, not a genuine result. The `urgency` classifier
  (5-fold CV ≈ 0.74, train−val gap ≈ 0.06) is the realistic task. Train vs
  validation accuracy is reported explicitly for both.
- **RUL cap = 125 cycles** (standard piecewise-linear RUL for CMAPSS) —
  early-life engines are intentionally trained toward a flat ceiling because
  early prognosis is not actionable.
- **Digital-twin economics** (throughput, downtime cost, MTTR, catastrophic
  cost) are documented constructor parameters in
  `src/digital_twin/simulator.py`; the *outcomes* are computed from the live
  `failure_prob` / `rul_estimate`, never hardcoded.

---

## 5. Preprocessing

**Tabular (`src/data/prepare_tabular.py`)**
1. Inject 180 duplicate rows + ~2.5 % missing values into 4 sensor columns
   (seeded) so handling can be demonstrated.
2. Drop exact + ID-level duplicates (before the split, so no row leaks across
   train/test).
3. Drop identifiers (`UDI`, `Product ID`) and hold out `TWF/HDF/PWF/OSF/RNF`
   (these are components of the target → leakage) for error analysis only.
4. Engineer `temp_diff_K`, `power_W`, `wear_torque`, one-hot `type_*`.
5. Stratified 60/20/20 split.
6. Median imputer + StandardScaler **fit on train only**, applied to val/test.

**CMAPSS** — RUL label = `min(max_cycle − cycle, 125)`; drop 7 constant sensors;
z-score using train statistics; sliding windows (best model: window 50).

**Images** — `image_dataset_from_directory`, resize 160², MobileNetV2
`preprocess_input`, light augmentation; train subset capped for CPU, full test
split always evaluated.

**Knowledge base** — PDF text extracted with `pypdf`, chunked by top-level
numbered section (23 chunks), embedded with `all-MiniLM-L6-v2`, FAISS cosine
index.

---

## 6. Results (headline)

| Modality | Model | Key metric |
|---|---|---|
| Tabular failure | RandomForest (best of RF → XGB → tuned XGB) | test F1 **0.76**, ROC-AUC **0.95** |
| RUL regression | LSTM (window 50) | test **MAE 11.2**, RMSE 14.9 cycles |
| Casting defect | MobileNetV2 (fine-tuned) | test F1 **0.96**, ROC-AUC **0.998** |
| Note category | TF-IDF + LogReg | 5-fold CV 1.00 *(trivial — see §4)* |
| Note urgency | TF-IDF + LogReg | 5-fold CV **0.74**, train−val gap 0.06 |

All runs logged to MLflow (`mlflow.db` + `mlruns/`), 3+ iterations per modality,
best model registered per modality (`factory_tabular_failure`,
`factory_cmapss_rul`, `factory_casting_defect`, `factory_nlp_category`,
`factory_nlp_urgency`). Full error analysis in `reports/`:
`tabular_error_analysis.md` (per-failure-mode recall; TWF recall = 0 with only 46
examples, RNF unlearnable by construction — both discussed),
`lstm_error_analysis.md`, `casting_cnn_report.md`, `nlp_report.md`,
`rag_report.md` (with- vs without-retrieval), `shap_tabular_report.md`.

---

## 7. Limitations

- Tabular positive class is 3.4 % — 68 test failures make F1 noisy; threshold is
  0.5 and not tuned per mode.
- RUL model trained on FD001 only (single operating condition); would not
  transfer to FD002–004 without retraining.
- Casting fine-tune run is sensitive to learning rate (a 1e-4 run collapsed;
  1e-5 is used) and the training set is CPU-capped at 2 400 images.
- NLP corpus is synthetic and templated (see §4).
- Knowledge base is 3 short synthetic documents; retrieval quality on a real
  document set is unverified.
- Digital-twin cost model is a first-order approximation, not a validated plant
  model.
- The LLM can still mis-phrase a citation label even though the structured
  `source_doc`/`source_section` returned to the UI is always correct.

---

## 8. Running it

```bash
pip install -r requirements.txt          # already satisfied in the grading env
# .env must contain: GROQ_API_KEY=...

python src/data/verify_setup.py           # Block 1  - data check
python src/data/prepare_tabular.py         # Block 2
python src/data/eda_tabular.py             # Block 2  - EDA plots
python src/models/train_tabular.py         # Block 3a + 10
python src/models/train_lstm_rul.py        # Block 3b + 10
python src/models/train_casting_cnn.py     # Block 4  + 10
python src/models/train_nlp.py             # Block 5  + 10
python src/data/make_knowledge_base.py     # Block 6  - build the 3 PDFs
python src/rag/build_index.py              # Block 6  - FAISS index
python src/rag/compare_retrieval.py        # Block 6  - with/without retrieval
python src/xai/shap_tabular.py             # Block 8
python src/xai/lstm_sensitivity.py         # Block 8
python src/tests/test_end_to_end.py        # Blocks 7 + 9 + 11 path

python -m streamlit run app/streamlit_app.py   # Block 11 - the console
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking
```

## 9. Deployment

The repo is deploy-ready. `.gitignore` excludes the 123 MB image corpus, the
MLflow store and secrets; the app falls back to `data/sample_images/` (80
committed casting photos) and reads `GROQ_API_KEY` from process env → `.env` →
`st.secrets`.

**Hugging Face Spaces (recommended — 16 GB RAM free, the full stack fits):**
1. Create a Space → SDK **Streamlit**.
2. Push this repo. Rename `requirements-deploy.txt` → `requirements.txt` in the
   Space (CPU-only TensorFlow) or add it as `requirements.txt`.
3. Space **Settings → Variables and secrets →** add `GROQ_API_KEY`.
4. Set the app file to `app/streamlit_app.py` (add to `README.md` front-matter or
   Space config: `app_file: app/streamlit_app.py`).

**Streamlit Community Cloud:** same repo; main file `app/streamlit_app.py`;
paste `GROQ_API_KEY` into *Secrets*. Note the free tier is **1 GB RAM** —
TensorFlow + FAISS + the loaded models run close to that ceiling; if it OOMs,
use Spaces.

```bash
git init && git add -A && git commit -m "AI Factory Command Center"
git remote add origin <your-repo>
git push -u origin main
```

Local model artifacts (`models_registry/*.keras`, `*.joblib`, `data/processed/rag/`)
**are committed** so the deployed app needs no training step. To regenerate them,
run the pipeline in §8.

## 10. Repo layout

```
data/{raw,processed,knowledge_base}   raw + cleaned data, FAISS index, decisions_log.csv
src/data/        organisation, EDA, tabular prep, KB generation
src/models/      tabular, LSTM, CNN, NLP trainers (each logs to MLflow)
src/xai/         SHAP (tabular) + permutation sensitivity (LSTM)
src/rag/         chunk → embed → FAISS → Groq generation, with citations
src/agents/      VisionAgent, PredictiveAgent, KnowledgeAgent, PlanningAgent, Orchestrator
src/digital_twin/ FactorySimulator (3 what-if scenarios)
src/report/      fpdf2 incident & decision report
app/streamlit_app.py   the console
models_registry/ deployed model artifacts + preprocessors
reports/         every error-analysis / XAI / EDA artifact
mlruns/, mlflow.db   MLflow tracking + model registry
```

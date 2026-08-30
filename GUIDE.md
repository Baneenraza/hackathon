# AI Factory Intelligence Command Center — Plain-English Guide

This guide explains what the project does, what every screen shows, what each of
the four "agents" is, and what every button does — with no jargon. If a technical
word is unavoidable, it's explained right where it appears.

---

## 1. What is this project, in one paragraph?

Imagine a factory control room with one screen. On that screen a supervisor can
check the health of a machine from three different angles at once:

1. **The machine's live sensor readings** (temperature, speed, torque, tool wear)
   → *is it about to break, and how?*
2. **The long-term wear history of rotating equipment** (like a jet engine)
   → *how many hours of life does it have left?*
3. **A photo of the part it just produced**
   → *is that part defective?*

The system also reads the **plant's maintenance manuals** and tells the
supervisor exactly what the official procedure says to do. Then it **simulates
the next week** under three different decisions (keep going, stop for
maintenance, or slow down) and shows the cost and risk of each.

Finally — and this is the important part — **it never acts on its own**. It
produces a recommendation, a human presses **APPROVE**, **REJECT**, or
**MODIFY**, and that decision is saved with the person's name and reason. A PDF
report of the whole incident can be downloaded.

Think of it as a very well-informed assistant that always hands the final
decision back to a person.

---

## 2. The "unified factory" story

The project uses three real public datasets that have nothing to do with each
other in real life. We *pretend* they are three stations on one imaginary
factory so the demo tells a single coherent story:

| Real dataset | In our story it is… |
|---|---|
| AI4I 2020 (machine sensor table) | the CNC machining line |
| NASA CMAPSS (jet-engine wear over time) | the rotating-equipment fleet |
| Casting product images | the quality-inspection camera |
| 450 maintenance notes + 3 PDF manuals (we wrote these) | the maintenance office |

The maintenance notes and manuals are **synthetic** (made up for the project),
but their numbers were written to match the real sensor data so everything lines
up. This is stated openly in the README — nothing is hidden.

---

## 3. The five failure types (you'll see these codes everywhere)

The machine data labels every breakdown as one of five kinds. In plain terms:

| Code | Full name | What actually goes wrong |
|---|---|---|
| **TWF** | Tool Wear Failure | the cutting tool is worn out and needs changing |
| **HDF** | Heat Dissipation Failure | the machine can't shed heat fast enough — it's overheating |
| **PWF** | Power Failure | the power draw is outside its safe band — the drive is struggling |
| **OSF** | Overstrain Failure | the tool is pushing too hard for too long — mechanical overload |
| **RNF** | Random Failure | a breakdown with no warning signs — bad luck, unpredictable |

---

## 4. The four agents (the "brains" behind the screen)

An **agent** here just means "a small self-contained helper that takes an input
and returns a tidy answer". There are four, and they run one after another.

### 4.1 VisionAgent — "the inspector"
- **Input:** a photo of a metal casting.
- **What it does:** a trained image model (MobileNetV2 — a standard, compact
  image-recognition network) looks at the photo and decides *defective* or *OK*.
- **Output:**
  - `defect` — yes or no
  - `severity` — how bad (0% = borderline, 100% = obvious defect)
  - `confidence` — how sure the model is
- **Bonus:** it also produces a **Grad-CAM** picture — the photo with a
  heat-map overlaid, showing *which pixels* made it say "defect". Red blobs on a
  real crack = the model is looking at the right thing. Red on a lighting
  reflection = the model was fooled, and a human should double-check.

### 4.2 PredictiveAgent — "the forecaster"
- **Input:** the machine's current sensor readings and/or an engine's wear
  history.
- **What it does:** runs **two** models:
  - a **RandomForest** classifier (a model that's basically a large committee of
    decision trees voting) → probability the machine fails *now*.
  - an **LSTM** (a neural network built for time-series / sequences) → the
    **RUL**, "Remaining Useful Life", i.e. estimated hours left before failure.
- **Output:** `failure_prob`, `rul_estimate`, `confidence`.

### 4.3 KnowledgeAgent — "the manual reader"
- **Input:** a question (e.g. "what do I do on an overstrain alert?") or just a
  failure-type code.
- **What it does:** this is **RAG** — "Retrieval-Augmented Generation". Two steps:
  1. **Retrieve:** search the three PDF manuals for the most relevant sections
     (using a text-similarity search — sentences with similar meaning score
     higher).
  2. **Generate:** hand those sections to a language model (Groq's
     `openai/gpt-oss-20b`) and ask it to phrase an answer **using only those
     sections**.
- **Output:** `answer`, plus **`source_doc` and `source_section`** so a human can
  open the exact page. If the manuals don't cover the question, it says so
  instead of guessing.
- **Key point:** the language model never predicts anything. It only rewrites
  what the manual already says into a clean sentence.

### 4.4 PlanningAgent — "the advisor"
- **Input:** the outputs of the other three agents.
- **What it does:**
  1. Combines them into a single **risk level**: LOW, MEDIUM, or HIGH.
  2. Runs the **digital twin** (`FactorySimulator`) — a small model of the
     factory's economics — for three scenarios (see §6).
  3. Writes a recommendation in plain words, including the manual's citation.
- **Output:** `recommendation`, `risk_level`, `simulated_outcomes` (the three
  scenarios), and **`requires_human_approval` — which is always `True`**.

### 4.5 The Orchestrator — "the conductor"
Not a brain of its own — it just calls the four agents in order
(Vision → Predictive → Knowledge → Planning) and collects everything into one
result. This is what the **Run full analysis** button triggers.

---

## 5. The screen, area by area

The app has **one page**. On the **left** is a grey sidebar for inputs; the
**main area** (right) shows results after you press the button.

### 5.1 Sidebar — "Incident inputs"

| Control | Plain meaning | Default |
|---|---|---|
| **Machine / asset ID** | a label for this machine, only used on the report | `CNC-L-02` |
| **Failure-mode hint** | if you already suspect a failure type (TWF/HDF/PWF/OSF/RNF), pick it — it tells the KnowledgeAgent which manual section to look up. Leave blank if unsure. | `OSF` |
| **or custom knowledge query** | instead of a code, type a full question for the manual reader | empty |

**"1. Process sensors (tabular)"** — the CNC machining line

| Control | Plain meaning |
|---|---|
| **Include tabular failure prediction** (checkbox) | tick to run the "will it fail now?" model |
| **Load random AI4I test row** (button) | fills the six boxes below with a real example row from the dataset, so you don't have to invent numbers |
| **Air temp [K]** | ambient air temperature, in Kelvin (~300 K = 27 °C) |
| **Process temp [K]** | temperature inside the machining process |
| **Rot. speed [rpm]** | spindle rotation speed |
| **Torque [Nm]** | how hard the spindle is twisting |
| **Tool wear [min]** | total minutes the current cutting tool has been used (new = 0, worn out ≈ 200–240) |
| **Type** | machine grade: **L** = light duty, **M** = medium, **H** = high-precision |

**"2. Turbofan sensor history (CMAPSS)"** — the rotating-equipment fleet

| Control | Plain meaning |
|---|---|
| **Include LSTM RUL estimate** (checkbox) | tick to run the "how many hours left?" model |
| **Test engine #** | pick one of 100 real test engines (1–100); the app feeds its recent sensor history to the model |

**"3. Casting image (vision)"** — the inspection camera

| Control | Plain meaning |
|---|---|
| **Upload a casting photo** | drop in your own JPG/PNG of a casting |
| **or use a sample** (radio) | `none` = skip vision; `defect sample` / `ok sample` = use one of the 80 built-in example photos |

### 5.2 Main area — before you run

- **Title + one-line description.**
- **Yellow banner:** the permanent reminder that the system is *decision support
  only* and a human holds authority. It also notes the manuals are synthetic.
- **`Run full analysis` button** (big, full-width): runs all four agents on
  whatever inputs you've set. First run takes ~30–60 s because the models load
  into memory once; later runs are quick.

### 5.3 Main area — after you run: "Results"

A line shows the **report ID** (e.g. `INC-20260830-153012`), the asset name, the
**agent chain** that ran, and how many seconds it took.

Then **three columns side by side:**

**Column 1 — VisionAgent**
- **Defect: YES / NO**
- severity and confidence percentages
- the **Grad-CAM image** (photo + heat-map of what the model looked at)
- *(if you didn't provide an image: "no image provided")*

**Column 2 — PredictiveAgent**
- **Failure probability** — a big percentage (chance the machine fails now)
- **RUL estimate** — hours of useful life remaining
- confidence
- **"Why (SHAP)"** — one sentence naming the top three readings that pushed the
  prediction up or down. *SHAP is a standard method that fairly splits a
  prediction into "how much did each input contribute".*
- a small **bar chart** of those contributions (red bar = pushed risk up,
  blue = pushed risk down)

**Column 3 — KnowledgeAgent**
- the **answer** pulled from the manuals
- a **green box** with the exact source: document name + section number + heading
- an expander **"retrieved passages"** — click to see all the manual sections it
  considered and their similarity scores
- *(if you asked nothing: "no query")*

**Below the columns — "PlanningAgent – recommendation"**
- a coloured box:
  - **red** = HIGH risk, **yellow** = MEDIUM, **blue** = LOW
  - shows the risk level, a numeric risk score, and the written recommendation
    (which includes the manual citation)

**"Digital-twin what-if"**
- a **table** with three rows — the three scenarios (see §6) — and columns:
  - `expected_units` — how many good parts you'd make over the next week
  - `downtime_hours` — expected hours the machine is stopped
  - `cost` — expected total cost (lost production + repairs + running cost)
  - `risk_score` — chance of an unplanned breakdown in that scenario
  - `net_value` — rough profit (value of parts made minus cost)
- a **chart**: bars = expected units per scenario, red line = risk of each
- a note saying which scenario the twin itself would pick
- **These numbers are calculated live** from the failure probability and RUL
  above — they are not typed in or faked.

### 5.4 Main area — "Human decision (required)"

This is the hand-off to a person.

| Control | Plain meaning |
|---|---|
| **Decision** (radio: APPROVE / REJECT / MODIFY) | **APPROVE** = do what the system recommends. **REJECT** = don't; keep running with extra monitoring. **MODIFY** = do something different (write what in the notes). |
| **Decided by** (text) | the name/role of the person making the call |
| **Reason / modification notes** (text box) | why they chose that — required for the audit trail |
| **`Log decision` button** | saves a row to `data/processed/decisions_log.csv` (timestamp, machine, prediction, risk, the decision, who, why). A green "Logged to…" confirmation appears. |

After you log a decision:

| Control | Plain meaning |
|---|---|
| **`Download incident & decision PDF` button** | builds a one/two-page PDF report of the whole incident — predictions, the "why", the manual guidance, the three simulated scenarios, and the human decision — and downloads it. Good for filing or emailing. |

At the bottom, an expander **"decision log"** shows the full history of every
decision logged so far, as a table.

---

## 6. The three digital-twin scenarios (what "what-if" means here)

The PlanningAgent asks: *given this failure probability and remaining life, what
happens over the next 7 days if we…*

| Scenario | Plain meaning | Typically… |
|---|---|---|
| **continue** | keep running the machine exactly as now | most parts made, but highest chance of a sudden expensive breakdown |
| **maintenance_stop** | stop now, do the planned fix (≈4 h), then resume | lose a few hours on purpose, but breakdown risk drops to near zero |
| **reduced_load** | keep running but slower / gentler | fewer parts, moderate risk — a middle option |

The simulator turns each into concrete numbers (units, downtime, cost, risk) so
the human can compare apples to apples instead of arguing from gut feel.

---

## 7. What happens behind the scenes when you press "Run full analysis"

1. The app gathers your sidebar inputs.
2. **Orchestrator** runs:
   - **VisionAgent** on the image (if any)
   - **PredictiveAgent** on the sensor values and engine history
   - **KnowledgeAgent** on your question / failure-mode hint
   - **PlanningAgent** on all of the above → risk level + 3 simulations +
     recommendation
3. Separately the app computes the **SHAP "why"** and the **Grad-CAM** picture.
4. Everything is shown on screen.
5. Nothing is sent anywhere and nothing on the factory changes. The only action
   the system takes is when *you* press **Log decision** (writes a CSV row) or
   **Download PDF** (writes a PDF).

---

## 8. The other parts of the project (not in the app)

You don't need these to use the console, but they're what built it:

| Folder / file | Plain purpose |
|---|---|
| `src/data/` | organises the raw data, cleans it, makes the EDA charts, writes the 3 synthetic manuals |
| `src/models/` | trains the four model types; every training run is recorded in **MLflow** (an experiment logbook — open it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`) |
| `src/xai/` | the "explainability" tools — SHAP for the sensor model, a sensitivity test for the engine model |
| `src/rag/` | builds the searchable index over the manuals |
| `src/agents/` | the four agents + orchestrator described in §4 |
| `src/digital_twin/` | the FactorySimulator from §6 |
| `src/report/` | the PDF builder |
| `reports/` | every analysis write-up: model accuracy, error analysis, the SHAP charts, the Grad-CAM grid, the "with vs without manual" comparison |
| `README.md` | the technical version of this guide |

---

## 9. The one rule to remember

**The AI only advises. A named human decides, and that decision is recorded.**
Every screen, every report, and the code itself is built around that.

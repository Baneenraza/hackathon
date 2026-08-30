# Hosting on Streamlit Community Cloud

Step-by-step. Takes ~10–15 min (most of it is the first build).

---

## 0. Prerequisites

- The code is pushed to GitHub: `https://github.com/Baneenraza/hackathon` (branch `main`).
- You have a free account at **https://share.streamlit.io** (sign in with GitHub).
- Your Groq API key (the value in your local `.env`, starting `gsk_...`).

The repo is already prepared:
- `requirements.txt` → the lean runtime set (uses `tensorflow-cpu`).
- `models_registry/` and `data/processed/rag/` are committed, so **no training
  runs on the server** — the app just loads the saved models.
- `data/sample_images/` (80 photos) ships in the repo, so the "use a sample"
  option works without the 123 MB image corpus.
- `.env` is git-ignored; the key goes in Streamlit **Secrets** instead.

---

## 1. Push the latest commit

```powershell
cd C:\Users\banee\OneDrive\Desktop\finalhackathon
git add -A
git commit -m "Deployment config: lean requirements, deploy guide"
git push -u origin main
```

---

## 2. Create the app on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** → **Create app** → **Deploy a public app
   from GitHub**.
2. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `Baneenraza/hackathon` |
   | Branch | `main` |
   | Main file path | `app/streamlit_app.py` |
   | App URL | pick anything, e.g. `ai-factory-command-center` |
3. Click **Advanced settings**:
   - **Python version:** `3.12` — **REQUIRED. Do not use 3.13 or 3.14.**
     TensorFlow has no wheels for 3.13+ and the build will fail. The repo also
     ships a `.python-version` file pinning 3.12, but set the dropdown too.
   - **Secrets:** paste this (TOML format), with your real key:
     ```toml
     GROQ_API_KEY = "gsk_your_real_key_here"
     ```
4. Click **Deploy**.

### Already created the app and it failed on `tensorflow`?

That means it built on Python 3.13/3.14. Fix without recreating:
1. `git push` the latest commit (adds `.python-version` = 3.12 and cleaner
   requirements).
2. In the app's page → **⋮ menu → Settings → General → Python version → `3.12`**
   → **Save**.
3. **⋮ menu → Reboot app.** It rebuilds from scratch on 3.12.

---

## 3. First build (~5–10 min)

The build log will show pip installing TensorFlow-CPU, PyTorch (pulled by
`sentence-transformers`), FAISS, etc. This is normal and only happens once.

When it finishes you'll see the app. On the **first run** it also downloads the
sentence-embedding model `all-MiniLM-L6-v2` (~90 MB) — the first "Run full
analysis" click will take ~30–60 s; later clicks are fast.

---

## 4. Smoke-test the live app

In the sidebar:
1. Leave the tabular defaults, tick **Include tabular failure prediction**.
2. Set **Test engine #** to `3`, tick **Include LSTM RUL estimate**.
3. Set **or use a sample** to `defect sample`.
4. Leave **Failure-mode hint** = `OSF`.
5. Press **Run full analysis**.

You should get: a defect verdict + Grad-CAM image, a failure probability + RUL,
a manual answer citing `safety_sop.pdf section 3.0`, a HIGH-risk recommendation,
and the 3-scenario digital-twin table. Then try **APPROVE** → **Log decision** →
**Download incident & decision PDF**.

---

## 5. If the app crashes with "Oh no." / restarts / runs out of memory

Community Cloud gives ~2.7 GB RAM. TensorFlow + PyTorch + the models sit around
1.5–2 GB, so it *usually* fits but can be tight. Options, easiest first:

- **Reboot the app** (top-right menu → *Reboot app*) — transient OOM often clears.
- **Uncheck modalities you're not demoing.** Each unchecked box skips loading
  that model. If you only need tabular + knowledge, untick the LSTM box.
- **Switch to Hugging Face Spaces** (free, 16 GB RAM — no tuning needed):
  - Create a Space, SDK = **Streamlit**, push the same repo.
  - Add `GROQ_API_KEY` under *Settings → Variables and secrets*.
  - In the Space's `README.md` front-matter set `app_file: app/streamlit_app.py`.

---

## 6. Updating the app later

Just `git push` to `main` — Streamlit redeploys automatically. Changing
`requirements.txt` triggers a full rebuild; changing only `.py` files is a fast
reload.

---

## 7. Notes for the graders / demo

- The knowledge base PDFs and maintenance notes are **synthetic** (stated in the
  README) — thresholds were written to line up with the AI4I sensor data.
- The Groq LLM only *phrases* retrieved manual text; it never predicts failure,
  RUL, or defects.
- Every recommendation ends at a human APPROVE / REJECT / MODIFY that is logged
  with a name and reason. The system has no autonomous authority.

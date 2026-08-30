"""Block 6 - retrieval-augmented generation over the knowledge base.

RAGPipeline.answer(query)            -> retrieval-grounded answer + citation
RAGPipeline.answer_no_retrieval(q)  -> ungrounded LLM answer (for the comparison)

Retrieval has two interchangeable back-ends over the same 23 section chunks:
  * "embed"  - sentence-transformers + FAISS cosine  (semantic; used locally)
  * "tfidf"  - scikit-learn TF-IDF + cosine          (keyword; used on deploy,
               where torch/faiss add ~2 GB and can segfault next to TensorFlow)
The pipeline auto-selects: it uses "embed" if sentence-transformers, faiss and a
prebuilt index are all present, otherwise "tfidf". Force with env RAG_BACKEND.

Every grounded answer cites source_doc + source_section. The LLM (Groq) only
phrases the retrieved text; if nothing scores above threshold the pipeline
refuses instead of guessing.
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

RAG_DIR = os.path.join(C.PROCESSED, "rag")
CHUNKS_PATH = os.path.join(RAG_DIR, "chunks.json")
FAISS_PATH = os.path.join(RAG_DIR, "faiss.index")
META_PATH = os.path.join(RAG_DIR, "meta.json")

# cosine thresholds below which we refuse (scales differ per back-end)
MIN_SCORE = {"embed": 0.25, "tfidf": 0.04}


def _embed_available():
    if os.getenv("RAG_BACKEND") == "tfidf":
        return False
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        return False
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class RAGPipeline:
    def __init__(self, backend=None):
        self.chunks = json.load(open(CHUNKS_PATH, encoding="utf-8"))
        self.backend = backend or ("embed" if _embed_available() else "tfidf")
        self._groq = None
        if self.backend == "embed":
            import faiss
            from sentence_transformers import SentenceTransformer
            self.meta = json.load(open(META_PATH))
            self.index = faiss.read_index(FAISS_PATH)
            self.emb = SentenceTransformer(self.meta["model"])
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.meta = {"model": "tfidf-word(1,2)", "n_chunks": len(self.chunks)}
            # keep numbers and short tokens ("H", "8.6", "250"); no stop-word list
            # (corpus is tiny); weight the section heading by repeating it
            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                        token_pattern=r"(?u)\b[\w.]+\b")
            docs = [f"{c['heading']} {c['heading']} {c['body']}" for c in self.chunks]
            self._mat = self._vec.fit_transform(docs)

    # ---------- retrieval ----------
    def retrieve(self, query, k=3):
        if self.backend == "embed":
            q = self.emb.encode([query], normalize_embeddings=True).astype("float32")
            scores, idx = self.index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self._vec.transform([query])
            sims = linear_kernel(qv, self._mat).ravel()
            idx = np.argsort(sims)[::-1][:k]
            scores = sims[idx]
        hits = []
        for s, i in zip(scores, idx):
            if i < 0:
                continue
            c = dict(self.chunks[int(i)])
            c["score"] = round(float(s), 4)
            hits.append(c)
        return hits

    # ---------- generation ----------
    def _groq_client(self):
        if self._groq is None:
            from groq import Groq
            self._groq = Groq(api_key=C.load_groq_key())
        return self._groq

    def _chat(self, system, user, max_tokens=600):
        r = self._groq_client().chat.completions.create(
            model=C.GROQ_MODEL, temperature=0.1, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content.strip()

    def answer(self, query, k=None):
        k = k or (3 if self.backend == "embed" else 4)
        hits = self.retrieve(query, k)
        if not hits or hits[0]["score"] < MIN_SCORE.get(self.backend, 0.1):
            return {"answer": "Not covered by the knowledge base - no supporting "
                              "section found. Escalate to a human engineer.",
                    "source_doc": None, "source_section": None,
                    "retrieved": hits, "grounded": False}
        context = "\n\n".join(
            f"({h['doc']} section {h['section']} - {h['heading']}) {h['body']}"
            for h in hits)
        system = ("You are a factory maintenance assistant. Answer ONLY from the "
                  "numbered context passages provided. Every passage begins with "
                  "its source in parentheses, e.g. (safety_sop.pdf section 3.0). "
                  "Cite the passage you used by copying that parenthesised source "
                  "exactly - never invent a document name or write placeholders. "
                  "If the context does not contain the answer, say so. Be concise "
                  "(max 4 sentences).")
        user = f"Context:\n{context}\n\nQuestion: {query}"
        text = self._chat(system, user)
        # cite the passage the answer actually used (the model names the section);
        # fall back to the top retrieved hit
        top = hits[0]
        for h in hits:
            if f"section {h['section']}" in text and h["doc"] in text:
                top = h
                break
        return {"answer": text, "source_doc": top["doc"],
                "source_section": top["section"], "source_heading": top["heading"],
                "retrieved": hits, "grounded": True, "backend": self.backend}

    def answer_no_retrieval(self, query):
        system = ("You are a factory maintenance assistant. Answer from general "
                  "knowledge. Be concise (max 4 sentences).")
        return {"answer": self._chat(system, query), "source_doc": None,
                "source_section": None, "retrieved": [], "grounded": False}


if __name__ == "__main__":
    rag = RAGPipeline()
    print("backend:", rag.backend)
    for q in ["What should I do when an overstrain OSF alert fires?",
              "How often should Type H machines get preventive maintenance?",
              "What is the acceptance tolerance for torque sensor calibration?"]:
        out = rag.answer(q)
        print("\nQ:", q)
        print("A:", out["answer"])
        print("cite:", out["source_doc"], out["source_section"],
              "| top score", out["retrieved"][0]["score"] if out["retrieved"] else None)

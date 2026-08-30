"""Block 6 - retrieval-augmented generation over the knowledge base.

RAGPipeline.answer(query)            -> retrieval-grounded answer + citation
RAGPipeline.answer_no_retrieval(q)  -> ungrounded LLM answer (for the comparison)

Every grounded answer cites source_doc + source_section. The LLM (Groq) is a
generation layer only: retrieval selects the evidence, the model just phrases it.
If retrieval finds nothing above threshold, the pipeline refuses rather than
guessing.
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
MIN_SCORE = 0.25


class RAGPipeline:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.meta = json.load(open(os.path.join(RAG_DIR, "meta.json")))
        self.chunks = json.load(open(os.path.join(RAG_DIR, "chunks.json")))
        self.index = faiss.read_index(os.path.join(RAG_DIR, "faiss.index"))
        self.emb = SentenceTransformer(self.meta["model"])
        self._groq = None

    # ---------- retrieval ----------
    def retrieve(self, query, k=3):
        q = self.emb.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        hits = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            c = dict(self.chunks[i])
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

    def answer(self, query, k=3):
        hits = self.retrieve(query, k)
        if not hits or hits[0]["score"] < MIN_SCORE:
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
        top = hits[0]
        return {"answer": text, "source_doc": top["doc"],
                "source_section": top["section"], "source_heading": top["heading"],
                "retrieved": hits, "grounded": True}

    def answer_no_retrieval(self, query):
        system = ("You are a factory maintenance assistant. Answer from general "
                  "knowledge. Be concise (max 4 sentences).")
        return {"answer": self._chat(system, query), "source_doc": None,
                "source_section": None, "retrieved": [], "grounded": False}


if __name__ == "__main__":
    rag = RAGPipeline()
    for q in ["What should I do when an overstrain OSF alert fires?",
              "How often should Type H machines get preventive maintenance?",
              "What is the acceptance tolerance for torque sensor calibration?"]:
        out = rag.answer(q)
        print("\nQ:", q)
        print("A:", out["answer"])
        print("cite:", out["source_doc"], out["source_section"],
              "| top score", out["retrieved"][0]["score"] if out["retrieved"] else None)

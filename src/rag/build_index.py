"""Block 6 - chunk the 3 knowledge-base PDFs by numbered section, embed, FAISS index.

Output -> data/processed/rag/
    chunks.json   list of {id, doc, section, heading, text, n_chars}
    faiss.index   normalized-embedding inner-product index (cosine similarity)
    meta.json     {model, dim, n_chunks}
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

OUT = os.path.join(C.PROCESSED, "rag")
os.makedirs(OUT, exist_ok=True)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCS = ["safety_sop.pdf", "maintenance_manual.pdf", "calibration_checklist.pdf"]

# top-level section heading: "3.0  Heading" at line start (sub-points N.1/N.2 stay
# inside their parent section's body)
HEAD_RE = re.compile(r"^(\d+\.0)\s+([A-Z].{3,70})$")


def read_pdf_text(path):
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def chunk_by_section(doc_name, text):
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks = []
    cur = None
    for ln in lines:
        m = HEAD_RE.match(ln.strip())
        if m:
            if cur:
                chunks.append(cur)
            cur = {"doc": doc_name, "section": m.group(1),
                   "heading": m.group(2).strip(), "text_lines": []}
        elif cur is not None:
            s = ln.strip()
            if not s:
                continue
            if re.search(r"-\s*page\s*\d+", s, re.I):      # running footer
                continue
            if "Synthetic reference document" in s or "demonstration only" in s:
                continue
            cur["text_lines"].append(s)
    if cur:
        chunks.append(cur)

    out = []
    for i, c in enumerate(chunks):
        body = " ".join(c["text_lines"]).strip()
        # strip the running footer text if it leaked in
        body = re.sub(r"\s*-\s*page \d+\s*", " ", body)
        full = f"[{c['doc']} section {c['section']} - {c['heading']}] {body}"
        out.append({"id": f"{c['doc']}::{c['section']}",
                    "doc": c["doc"], "section": c["section"],
                    "heading": c["heading"], "text": full,
                    "body": body, "n_chars": len(body)})
    return out


def main():
    all_chunks = []
    for d in DOCS:
        path = os.path.join(C.KNOWLEDGE_BASE, d)
        text = read_pdf_text(path)
        ch = chunk_by_section(d, text)
        print(f"{d}: {len(ch)} sections")
        for c in ch:
            print(f"   {c['section']:5s} {c['heading'][:50]:50s} {c['n_chars']:4d} chars")
        all_chunks += ch

    print(f"\ntotal chunks: {len(all_chunks)}")
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode([c["text"] for c in all_chunks], normalize_embeddings=True,
                       show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(OUT, "faiss.index"))
    json.dump(all_chunks, open(os.path.join(OUT, "chunks.json"), "w"), indent=2)
    json.dump({"model": EMB_MODEL, "dim": int(emb.shape[1]),
               "n_chunks": len(all_chunks)},
              open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"index + chunks -> {os.path.relpath(OUT, C.ROOT)}")
    print("BLOCK 6 (index) OK")


if __name__ == "__main__":
    main()

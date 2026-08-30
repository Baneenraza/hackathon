"""KnowledgeAgent - retrieval-augmented answer over the knowledge base.

run(query, context) -> {"answer": str, "source_doc": str, "source_section": str}
`context` (optional str) is appended to the query to steer retrieval, e.g. the
current failure mode or machine type coming from the other agents.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.rag import RAGPipeline


class KnowledgeAgent:
    def __init__(self, pipeline=None):
        self.rag = pipeline or RAGPipeline()

    def run(self, query, context=""):
        q = f"{query}\nContext: {context}".strip() if context else query
        out = self.rag.answer(q)
        top = out["retrieved"][0] if out["retrieved"] else {}
        return {
            "answer": out["answer"],
            "source_doc": out.get("source_doc"),
            "source_section": out.get("source_section"),
            "source_heading": top.get("heading"),
            "grounded": out["grounded"],
            "retrieved": [{"doc": h["doc"], "section": h["section"],
                           "heading": h["heading"], "score": h["score"]}
                          for h in out["retrieved"]],
        }


if __name__ == "__main__":
    ka = KnowledgeAgent()
    print(ka.run("What do I do on an overstrain alert?", context="failure mode OSF, Type H"))

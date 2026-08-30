# Block 6 - RAG: retrieval-grounded vs ungrounded generation

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` | chunks: 23 | LLM: `openai/gpt-oss-20b` (Groq) | retriever: FAISS cosine, top-3

## Q: At what tool-wear time must we schedule a tool change, and what is the hard limit?

**WITH retrieval (grounded):**
> Schedule a tool change at 200 minutes of cumulative wear (safety_sop.pdf section 5.0). The hard limit is 240 minutes; a tool must never run past this point (safety_sop.pdf section 5.0).

- cited source: `maintenance_manual.pdf` section 3.0 (Tool Wear Management (TWF))
- retrieved: maintenance_manual.pdf#3.0 (0.6899), calibration_checklist.pdf#6.0 (0.568), safety_sop.pdf#5.0 (0.4699)

**WITHOUT retrieval (ungrounded LLM):**
> Schedule a tool change when wear reaches about 80 % of its allowable life.  
The hard limit is the point of 100 % wear, beyond which the tool must be replaced immediately.  
Continuing to use a tool past this limit risks failure and safety hazards.

- no source, no plant-specific thresholds - generic and unverifiable.

---

## Q: What air-to-process temperature difference indicates heat-dissipation risk?

**WITH retrieval (grounded):**
> Heat‑dissipation risk occurs when the air‑to‑process temperature difference falls below 8.6 K while the machine is running under 1380 rpm. (safety_sop.pdf section 4.0)

- cited source: `safety_sop.pdf` section 4.0 (Response to Heat Dissipation (HDF) Conditions)
- retrieved: safety_sop.pdf#4.0 (0.4974), calibration_checklist.pdf#3.0 (0.4861), calibration_checklist.pdf#2.0 (0.4057)

**WITHOUT retrieval (ungrounded LLM):**
> A small temperature differential between the ambient air and the process fluid signals a heat‑dissipation risk. In most industrial settings, a difference of less than about 10 °C (18 °F) is considered too low and can lead to overheating. If the gap drops below 5 °C (9 °F), the risk becomes critical and immediate cooling action is required.

- no source, no plant-specific thresholds - generic and unverifiable.

---

## Q: Who has to approve a HIGH risk maintenance recommendation?

**WITH retrieval (grounded):**
> A HIGH‑risk recommendation must be signed off by both the maintenance supervisor and the shift engineer. (safety_sop.pdf section 6.0 - Human Approval and Escalation)

- cited source: `safety_sop.pdf` section 6.0 (Human Approval and Escalation)
- retrieved: safety_sop.pdf#6.0 (0.6303), safety_sop.pdf#1.0 (0.4441), maintenance_manual.pdf#6.0 (0.4227)

**WITHOUT retrieval (ungrounded LLM):**
> High‑risk maintenance recommendations must be approved by the Safety Manager (or equivalent safety authority) and the Maintenance Manager (or Plant Manager). This dual approval ensures both safety compliance and operational feasibility.

- no source, no plant-specific thresholds - generic and unverifiable.

---

## Takeaway
- The grounded answers reproduce the plant's actual numbers (200 min / 240 min tool life, 8.6 K temperature delta, supervisor + shift engineer sign-off) and point to the exact section a human can open.
- The ungrounded answers are fluent and roughly sensible but cite nothing, invent no specific limits, and would not survive an audit.
- In the Command Center the LLM is never the predictor - retrieval selects the evidence and the LLM only phrases it, with a refusal path when nothing relevant is found.

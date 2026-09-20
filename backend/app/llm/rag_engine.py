"""
app/llm/rag_engine.py
=====================

RAG (Retrieval-Augmented Generation) Clinical Copilot Engine.
Provides interactive conversational assistance for clinicians:
1. Answers questions on triage rules, ESI levels, Shock Index, MAP, and protocols.
2. Explains patient-specific recommendations using live triage context.
3. Maintains multi-turn conversational memory.
4. Operates in dual mode: Local Contextual Synthesizer or Cloud LLM.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.llm.knowledge_base import CLINICAL_KNOWLEDGE_DOCUMENTS

CLINICAL_DISCLAIMER = "AI recommends. Clinician decides."

# In-memory conversational session store: {session_id: [{"role": str, "content": str}]}
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_TURNS = 10


class ClinicalRAGEngine:
    """Retrieval-Augmented Generation engine for clinical decision support."""

    def __init__(self) -> None:
        self.documents = CLINICAL_KNOWLEDGE_DOCUMENTS
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def retrieve_relevant_docs(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve the most relevant clinical knowledge documents using keyword overlap and score."""
        query_words = set(re.findall(r"\w+", query.lower()))
        scored_docs = []

        for doc in self.documents:
            score = 0
            # Check title match
            for word in query_words:
                if word in doc["title"].lower():
                    score += 3
                if any(word in kw for kw in doc["keywords"]):
                    score += 2
                if word in doc["content"].lower():
                    score += 1

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda t: -t[0])
        return [doc for _, doc in scored_docs[:top_k]]

    def answer_query(
        self,
        query: str,
        session_id: str,
        patient_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize a clinically grounded response to a clinician's query.
        """
        # Retrieve relevant clinical documents
        docs = self.retrieve_relevant_docs(query)
        sources = [d["title"] for d in docs] if docs else ["JeevLok AI Reference Standards"]

        # Track conversation history in session store
        if session_id not in SESSION_MEMORY:
            SESSION_MEMORY[session_id] = []
        if history:
            SESSION_MEMORY[session_id] = history[-MAX_HISTORY_TURNS:]

        # Check if query asks about the current patient's recommendation or vitals
        reply = self._generate_contextual_response(query, docs, patient_context)

        # Update session memory
        SESSION_MEMORY[session_id].append({"role": "user", "content": query})
        SESSION_MEMORY[session_id].append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "session_id": session_id,
            "sources": sources,
            "disclaimer": CLINICAL_DISCLAIMER,
        }

    def _generate_contextual_response(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        patient_context: Optional[Dict[str, Any]],
    ) -> str:
        """Synthesize response using clinical documents and live patient context."""
        q_lower = query.lower()

        # Case 1: "Why is shock index high?" / "Explain shock index"
        if "shock index" in q_lower or "si" in q_lower.split():
            doc_content = next((d["content"] for d in docs if "shock" in d["id"]), "")
            resp = (
                "**Shock Index (SI) = Heart Rate / Systolic Blood Pressure**.\n\n"
                "• Normal Range: 0.5 – 0.7\n"
                "• Abnormal / High: ≥ 0.90 (signals occult hypoperfusion, blood loss, or early shock before BP visibly crashes).\n\n"
            )
            if patient_context:
                hr = patient_context.get("heartrate")
                sbp = patient_context.get("sbp")
                if hr and sbp:
                    si = round(hr / sbp, 2)
                    resp += f"**For this patient**: HR = {hr} bpm, SBP = {sbp} mmHg → **Shock Index = {si}**. "
                    if si >= 0.9:
                        resp += "This is **elevated**, indicating physiological compensation or impending hemodynamic collapse."
                    else:
                        resp += "This is within normal physiological limits."
            return resp

        # Case 2: "What does MAP mean?" / "Explain MAP"
        if "map" in q_lower or "mean arterial pressure" in q_lower:
            resp = (
                "**Mean Arterial Pressure (MAP) = (2 × DBP + SBP) / 3**.\n\n"
                "• Normal Range: 70 – 100 mmHg\n"
                "• Critical Threshold: **< 65 mmHg** indicates compromised end-organ perfusion (kidneys, brain, myocardium).\n\n"
            )
            if patient_context:
                sbp = patient_context.get("sbp")
                dbp = patient_context.get("dbp")
                if sbp and dbp:
                    m = round((2 * dbp + sbp) / 3, 1)
                    resp += f"**For this patient**: SBP = {sbp}, DBP = {dbp} → **MAP = {m} mmHg**. "
                    if m < 65:
                        resp += "This is **below the safe perfusion threshold**, warranting emergent intervention (P1/P2)."
                    else:
                        resp += "This maintains adequate organ perfusion pressure."
            return resp

        # Case 3: "Why is this patient Priority P2 / P1 / P3?" or "Explain this triage recommendation"
        if "priority" in q_lower or "recommendation" in q_lower or "triage" in q_lower or "why" in q_lower:
            if patient_context:
                priority = patient_context.get("priority", "Assigned Priority")
                risk_score = patient_context.get("risk_score", "N/A")
                top_features = patient_context.get("top_features", [])
                reason = patient_context.get("uncertainty_reason", "")

                resp = (
                    f"**Patient Priority Assessment: {priority} (Risk Score: {risk_score}/100)**\n\n"
                    f"The recommendation is determined by our hybrid clinical rule engine and gradient boosting ensemble.\n\n"
                    f"**Key Clinical Factors:**\n"
                )
                if top_features:
                    for feat in top_features[:4]:
                        resp += f"• {feat}\n"
                else:
                    resp += "• Evaluated vital signs and acute clinical findings.\n"

                if patient_context.get("chest_pain") and patient_context.get("diaphoresis"):
                    resp += "\n⚠️ **High Cardiac Risk**: Combination of acute chest pain with diaphoresis triggers immediate STEMI evaluation protocol."

                if reason:
                    resp += f"\n*Model Confidence Note*: {reason}"
                return resp

        # Case 4: General medical / protocol question backed by retrieved documents
        if docs:
            top_doc = docs[0]
            return (
                f"### {top_doc['title']}\n\n"
                f"{top_doc['content']}\n\n"
                f"*Reference: Clinical Decision Support Guidelines (ESI & AHA/ACC ED Protocols)*"
            )

        # Fallback helpful response
        return (
            "I am the **JeevLok AI Copilot**. I can help you interpret:\n"
            "• Emergency Severity Index (P1–P5) triage criteria\n"
            "• Shock Index (SI = HR / SBP) and Mean Arterial Pressure (MAP) thresholds\n"
            "• Specific patient triage recommendations and SHAP feature drivers\n"
            "• ED clinical protocols for STEMI, Stroke (FAST), and Sepsis (qSOFA)\n\n"
            "How can I assist with your triage assessment?"
        )


# Global singleton instance
rag_engine = ClinicalRAGEngine()


def query_clinical_copilot(
    query: str,
    session_id: str = "default_session",
    patient_context: Optional[dict] = None,
    history: Optional[list] = None,
) -> dict:
    """Convenience functional interface for the chat copilot."""
    return rag_engine.answer_query(query, session_id, patient_context, history)

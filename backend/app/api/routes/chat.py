"""
api/routes/chat.py
==================

Intelligent Chat Assistant (Clinical Copilot) Endpoint.
------------------------------------------------------
POST /api/v1/chat

Provides RAG-grounded conversational assistance for emergency nurses and clinicians:
- Answers queries about ESI triage criteria, Shock Index, MAP, and emergency protocols.
- Explains patient-specific AI triage recommendations and contributing SHAP features.
- Adheres to clinical safety boundaries with 'AI recommends. Clinician decides.'
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.logging_config import get_logger
from app.llm.rag_engine import query_clinical_copilot
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post("", response_model=ChatResponse)
def chat_with_copilot(body: ChatRequest) -> ChatResponse:
    """
    Query the Clinical Copilot assistant with optional patient context
    and multi-turn conversation history.
    """
    try:
        history_dicts = [m.model_dump() for m in body.history] if body.history else None
        res = query_clinical_copilot(
            query=body.message,
            session_id=body.session_id,
            patient_context=body.patient_context,
            history=history_dicts,
        )
        return ChatResponse(**res)
    except Exception as exc:
        logger.error("Failed to process copilot chat query: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clinical Copilot error: {str(exc)}")

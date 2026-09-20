"""
schemas/chat.py
===============

Pydantic schemas for the Intelligent Chat Assistant (Clinical Copilot).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single message in a conversational session."""
    role: str = Field(..., examples=["user", "assistant", "system"])
    content: str = Field(..., examples=["Why is shock index high?"])


class ChatRequest(BaseModel):
    """Clinical Copilot query with session and patient context."""
    message: str = Field(..., min_length=1, max_length=2000, examples=["Explain this triage recommendation"])
    session_id: str = Field(default="default_session", examples=["nurse_session_001"])
    patient_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional live patient vitals, priority, risk score, and findings to ground the AI response."
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional recent message history for multi-turn conversational context."
    )


class ChatResponse(BaseModel):
    """Grounded clinical response with citations and disclaimer."""
    reply: str
    session_id: str
    sources: List[str]
    disclaimer: str = "AI recommends. Clinician decides."

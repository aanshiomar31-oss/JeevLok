"""
schemas/summary.py
==================

Pydantic schemas for LLM Clinical Summary endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    """Request payload for generating an explainable clinical summary."""
    patient_data: Dict[str, Any] = Field(
        ...,
        description="Patient demographics, vitals, and findings (matches TriageRequest or triage_stay)."
    )
    triage_result: Dict[str, Any] = Field(
        ...,
        description="Hybrid ML prediction output (priority, risk_score, confidence, top_features, etc.)."
    )


class SummaryResponse(BaseModel):
    """Structured clinical summary with actionable rationale and disclaimer."""
    priority: str
    priority_description: str
    risk_score: int
    confidence: float
    summary_headline: str
    clinical_synthesis: str
    urgency_drivers: List[str]
    suggested_protocols: List[str]
    vitals_summary: Dict[str, Any]
    disclaimer: str = "AI recommends. Clinician decides."

"""
schemas/nlp.py
==============

Pydantic schemas for the Clinical NLP and Entity Extraction endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SymptomParseRequest(BaseModel):
    """Free-text clinical input to be parsed by the NLP pipeline."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["54yo male with crushing chest pain for 30 minutes, sweating and difficulty breathing. Denies fever. BP 90/60, HR 115."],
        description="Free-text clinical triage note, chief complaint, or transcribed voice dictation."
    )


class EntityChip(BaseModel):
    """Extracted entity chip for UI visualization."""
    label: str
    category: str
    is_negated: bool = False
    confidence: float = 1.0
    type: str = Field(..., description="'finding', 'negation', or 'vital'")


class ICDMapping(BaseModel):
    """Clinical concept normalized to ICD-10-CM."""
    concept_id: str
    standard_name: str
    category: str
    icd10: str
    icd10_desc: str
    matched_term: str
    similarity_score: float = 1.0
    method: str = "exact"


class SymptomParseResponse(BaseModel):
    """Full structured output of the Clinical NLP pipeline."""
    raw_text: str
    cleaned_text: str
    urgency_level: str = Field(..., description="'CRITICAL', 'HIGH', 'MODERATE', or 'LOW'")
    urgency_keywords: List[str] = Field(default_factory=list)
    vitals: Dict[str, Any] = Field(default_factory=dict)
    findings: Dict[str, bool] = Field(default_factory=dict)
    negated_findings: List[str] = Field(default_factory=list)
    entity_chips: List[EntityChip] = Field(default_factory=list)
    icd_mappings: List[ICDMapping] = Field(default_factory=list)
    auto_fill_payload: Dict[str, Any] = Field(..., description="Payload ready for auto-filling the triage form")


class NormalizeRequest(BaseModel):
    """Request to normalize a single symptom phrase into a standardized medical concept."""
    query: str = Field(..., min_length=1, max_length=200, examples=["heart attack feeling"])


class NormalizeResponse(BaseModel):
    """Normalized concept and ICD-10 code."""
    query: str
    matched: Optional[ICDMapping] = None

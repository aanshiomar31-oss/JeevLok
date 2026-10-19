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


class LLMTriageRequest(BaseModel):
    """Request for LLM-powered Patient Triage AI inference pipeline."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=[
            "58-year-old male with severe crushing substernal chest pain radiating to left jaw for 45 minutes, diaphoresis and shortness of breath. Denies fever. BP 90/60, HR 115, SpO2 89%."
        ],
        description="Unstructured clinical notes, transcribed triage voice intake, or free-text symptom description."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for LLM inference (0.0 for deterministic clinical classification)."
    )
    include_prompt_debug: bool = Field(
        default=True,
        description="Whether to include LangChain prompt orchestration and token inspection diagnostics."
    )


class VitalsSummary(BaseModel):
    """Physiological vitals and derived emergency hemodynamic indices."""
    age: Optional[float] = None
    gender: Optional[str] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    temperature: Optional[float] = None
    o2sat: Optional[float] = None
    pain: Optional[float] = None
    shock_index: Optional[float] = Field(None, description="Shock Index = HR / SBP (>=0.90 indicates occult shock)")
    map: Optional[float] = Field(None, description="Mean Arterial Pressure = (2*DBP + SBP)/3 (<65 indicates organ hypoperfusion)")
    qsofa_score: Optional[int] = Field(None, description="Quick SOFA score (>=2 indicates high risk sepsis)")


class PromptOrchestrationDebug(BaseModel):
    """LangChain prompt orchestration inspection and execution diagnostics."""
    system_prompt_preview: str
    orchestrated_human_prompt: str
    model_name: str
    framework: str = "LangChain LCEL"
    latency_ms: float
    fallback_used: bool


class LLMTriageResponse(BaseModel):
    """
    Structured Clinical Decision Support response from the LLM-powered Patient Triage AI pipeline.
    Combines NLP parsing, NegEx, LangChain prompt orchestration, ESI prioritization,
    and protocol triggers.
    """
    priority: str = Field(..., description="ESI Priority Level: 'P1', 'P2', 'P3', 'P4', or 'P5'")
    urgency_level: str = Field(..., description="'CRITICAL', 'HIGH', 'MODERATE', 'LOW', or 'NON-URGENT'")
    risk_score: float = Field(..., description="Estimated clinical acuity risk score (0-100)")
    confidence: float = Field(..., description="Model confidence score (0.0 - 1.0)")
    summary_headline: str = Field(..., description="Concise clinician-facing priority heading")
    clinical_rationale: str = Field(..., description="Synthesized clinical justification and reasoning")
    physiological_drivers: List[str] = Field(default_factory=list, description="Vital signs and shock/MAP thresholds driving urgency")
    differential_considerations: List[str] = Field(default_factory=list, description="Top differential diagnostic considerations")
    recommended_protocols: List[str] = Field(default_factory=list, description="Actionable ED clinical protocols (e.g. STEMI, Stroke FAST, Sepsis)")
    red_flags: List[str] = Field(default_factory=list, description="Immediate critical red flag indicators identified")
    vitals_summary: VitalsSummary = Field(..., description="Parsed vitals and calculated physiological indices")
    affirmed_findings: List[str] = Field(default_factory=list, description="Affirmed clinical findings after NegEx filtering")
    negated_findings: List[str] = Field(default_factory=list, description="Explicitly denied/negated symptoms")
    auto_fill_payload: Dict[str, Any] = Field(..., description="Form payload ready to auto-populate the patient intake queue")
    prompt_orchestration: Optional[PromptOrchestrationDebug] = None
    disclaimer: str = Field(default="AI recommends. Clinician decides.", description="Governing safety rule")


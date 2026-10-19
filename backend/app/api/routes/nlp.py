"""
api/routes/nlp.py
=================

Clinical NLP & Medical Entity Recognition Endpoints.
---------------------------------------------------
POST /api/v1/nlp/parse-symptoms
POST /api/v1/nlp/normalize

Extracts clinical findings, vitals, negations, urgency levels, and ICD-10
mappings from free-text notes, returning structured JSON to auto-fill the triage form.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.logging_config import get_logger
from app.nlp.pipeline import nlp_pipeline
from app.nlp.similarity import ClinicalSemanticMatcher
from app.llm.triage_chain import llm_triage_pipeline
from app.schemas.nlp import (
    SymptomParseRequest,
    SymptomParseResponse,
    NormalizeRequest,
    NormalizeResponse,
    EntityChip,
    ICDMapping,
    LLMTriageRequest,
    LLMTriageResponse,
    VitalsSummary,
    PromptOrchestrationDebug,
)

router = APIRouter(prefix="/nlp", tags=["nlp"])
logger = get_logger(__name__)
semantic_matcher = ClinicalSemanticMatcher()


@router.post("/parse-symptoms", response_model=SymptomParseResponse)
def parse_symptoms(body: SymptomParseRequest) -> SymptomParseResponse:
    """
    Parse free-text clinical symptoms into structured triage features.
    Extracts vitals, affirms findings, identifies negations (e.g. 'no chest pain'),
    classifies urgency level, and maps to ICD-10 codes.
    """
    try:
        result = nlp_pipeline.process_clinical_note(body.text)
        return SymptomParseResponse(
            raw_text=result["raw_text"],
            cleaned_text=result["cleaned_text"],
            urgency_level=result["urgency_level"],
            urgency_keywords=result["urgency_keywords"],
            vitals=result["vitals"],
            findings=result["findings"],
            negated_findings=result["negated_findings"],
            entity_chips=[EntityChip(**chip) for chip in result["entity_chips"]],
            icd_mappings=[ICDMapping(**m) for m in result["icd_mappings"]],
            auto_fill_payload=result["auto_fill_payload"],
        )
    except Exception as exc:
        logger.error("Failed to parse symptoms with NLP pipeline: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"NLP parsing error: {str(exc)}")


@router.post("/normalize", response_model=NormalizeResponse)
def normalize_symptom(body: NormalizeRequest) -> NormalizeResponse:
    """
    Normalize a colloquial symptom term (e.g., 'heart attack feeling')
    into its standardized clinical concept and ICD-10-CM code.
    """
    try:
        matched = semantic_matcher.match_concept(body.query)
        if matched:
            return NormalizeResponse(
                query=body.query,
                matched=ICDMapping(**matched),
            )
        return NormalizeResponse(query=body.query, matched=None)
    except Exception as exc:
        logger.error("Failed to normalize symptom '%s': %s", body.query, exc)
        raise HTTPException(status_code=500, detail=f"Normalization error: {str(exc)}")


@router.post("/llm-triage", response_model=LLMTriageResponse)
def llm_triage(body: LLMTriageRequest) -> LLMTriageResponse:
    """
    LLM-powered Patient Triage AI inference pipeline using LangChain prompt orchestration.
    Parses patient symptom descriptions and clinical context, prioritizing case urgency
    via a modular inference pipeline combining NLP preprocessing, NegEx, physiological vitals,
    and clinical emergency protocols.
    """
    try:
        result = llm_triage_pipeline.run_pipeline(
            raw_text=body.text,
            temperature=body.temperature,
            include_prompt_debug=body.include_prompt_debug,
        )
        return LLMTriageResponse(
            priority=result["priority"],
            urgency_level=result["urgency_level"],
            risk_score=result["risk_score"],
            confidence=result["confidence"],
            summary_headline=result["summary_headline"],
            clinical_rationale=result["clinical_rationale"],
            physiological_drivers=result["physiological_drivers"],
            differential_considerations=result["differential_considerations"],
            recommended_protocols=result["recommended_protocols"],
            red_flags=result["red_flags"],
            vitals_summary=VitalsSummary(**result["vitals_summary"]),
            affirmed_findings=result["affirmed_findings"],
            negated_findings=result["negated_findings"],
            auto_fill_payload=result["auto_fill_payload"],
            prompt_orchestration=PromptOrchestrationDebug(**result["prompt_orchestration"]) if result.get("prompt_orchestration") else None,
            disclaimer=result["disclaimer"],
        )
    except Exception as exc:
        logger.error("Failed to run LLM triage pipeline: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM triage pipeline error: {str(exc)}")


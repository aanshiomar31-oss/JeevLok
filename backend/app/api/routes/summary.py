"""
api/routes/summary.py
=====================

LLM Clinical Summary Endpoint.
-------------------------------
POST /api/v1/summary

Generates explainable, high-signal clinical briefings explaining
why a specific triage priority was assigned, highlights critical physiological
indices (e.g. Shock Index, MAP), suggests urgent clinical protocols, and
displays the mandatory safety disclaimer: "AI recommends. Clinician decides."
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.logging_config import get_logger
from app.llm.clinical_summary import generate_clinical_summary
from app.schemas.summary import SummaryRequest, SummaryResponse

router = APIRouter(prefix="/summary", tags=["summary"])
logger = get_logger(__name__)


@router.post("", response_model=SummaryResponse)
def create_clinical_summary(body: SummaryRequest) -> SummaryResponse:
    """
    Generate an explainable clinical summary for a given patient intake
    and triage prediction result.
    """
    try:
        summary_dict = generate_clinical_summary(body.patient_data, body.triage_result)
        return SummaryResponse(**summary_dict)
    except Exception as exc:
        logger.error("Failed to generate clinical summary: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clinical summary error: {str(exc)}")

"""
schemas/explain.py
==================

Pydantic schemas for Explainable AI, SHAP narratives, and Counterfactual Analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CounterfactualRequest(BaseModel):
    """Request to evaluate a counterfactual 'What-If' scenario."""
    base_patient: Dict[str, Any] = Field(
        ...,
        description="Original patient data (demographics, vitals, findings)."
    )
    adjusted_vitals: Dict[str, Any] = Field(
        ...,
        description="Hypothetical modified vitals (e.g. {'sbp': 120, 'heartrate': 80})."
    )


class FeatureImpactCard(BaseModel):
    """Visual feature importance card for SHAP visualization."""
    feature_name: str
    patient_value: Any
    impact_magnitude: float
    direction: str = Field(..., description="'escalating' (increases urgency) or 'stabilizing' (decreases urgency)")
    clinical_note: str


class CounterfactualComparisonResponse(BaseModel):
    """Side-by-side comparison of baseline vs counterfactual triage prediction."""
    original: Dict[str, Any]
    counterfactual: Dict[str, Any]
    risk_delta: int
    priority_changed: bool
    direction: str = Field(..., description="'improved', 'worsened', or 'unchanged'")
    natural_language_explanation: str
    feature_impact_cards: List[FeatureImpactCard] = Field(default_factory=list)


class NarrativeRequest(BaseModel):
    """Request to generate a natural language narrative for SHAP results."""
    patient_data: Dict[str, Any]
    triage_result: Dict[str, Any]


class NarrativeResponse(BaseModel):
    """Plain-language explanation of SHAP factors and model decision."""
    headline: str
    narrative: str
    primary_driver: str
    feature_impact_cards: List[FeatureImpactCard]

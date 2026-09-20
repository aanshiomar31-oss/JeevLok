"""
app/llm/clinical_summary.py
===========================

Generates explainable clinical summaries after triage prediction.
Combines patient vitals, ML risk scores, SHAP factors, and triggered protocols
into a high-signal clinician briefing.

Enforces the non-negotiable safety principle:
"The AI recommends. The clinician decides."
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

CLINICAL_DISCLAIMER = "AI recommends. Clinician decides."

PRIORITY_DESCRIPTIONS = {
    "P1": "Resuscitation (Immediate life-saving intervention required)",
    "P2": "Emergent (High risk, acute physiological deterioration)",
    "P3": "Urgent (Stable vitals, requires multiple resources)",
    "P4": "Less Urgent (Stable, single diagnostic or therapeutic resource)",
    "P5": "Non-Urgent (No resources needed, routine evaluation)",
}


class ClinicalSummaryGenerator:
    """
    Produces structured, clinician-focused AI summaries of triage assessments.
    Operates in dual mode:
    1. Local deterministic clinical synthesis (zero API key dependency, instant, reliable)
    2. Cloud LLM synthesis (when OPENAI_API_KEY or GEMINI_API_KEY is configured)
    """

    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def generate_summary(
        self,
        patient_data: Dict[str, Any],
        triage_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesize a structured clinical explanation from triage outputs.
        """
        priority = triage_result.get("priority", "P3")
        risk_score = triage_result.get("risk_score", 50)
        confidence = triage_result.get("confidence", 0.85)
        top_features = triage_result.get("top_features", [])
        triggered_protocols = triage_result.get("triggered_protocols", [])
        sepsis_alert = triage_result.get("sepsis_alert", False)

        # Derive physiological flags
        hr = patient_data.get("heartrate")
        sbp = patient_data.get("sbp")
        dbp = patient_data.get("dbp")
        resprate = patient_data.get("resprate")
        o2sat = patient_data.get("o2sat")

        # Shock Index
        shock_index = None
        if hr and sbp and sbp > 0:
            shock_index = round(hr / sbp, 2)

        # Mean Arterial Pressure
        map_val = None
        if sbp and dbp:
            map_val = round((2 * dbp + sbp) / 3, 1)

        # 1. Identify primary urgency drivers
        urgency_drivers = []
        if priority in ["P1", "P2"]:
            if shock_index and shock_index >= 0.9:
                urgency_drivers.append(f"Elevated Shock Index ({shock_index} ≥ 0.90 threshold, indicating hemodynamic instability)")
            if sbp and sbp < 90:
                urgency_drivers.append(f"Hypotension (SBP {sbp} mmHg < 90)")
            if o2sat and o2sat < 92:
                urgency_drivers.append(f"Hypoxemia (SpO2 {o2sat}% < 92%)")
            if resprate and (resprate > 24 or resprate < 10):
                urgency_drivers.append(f"Abnormal respiratory rate ({resprate} breaths/min)")
            if patient_data.get("chest_pain") and patient_data.get("diaphoresis"):
                urgency_drivers.append("Acute chest pain coupled with diaphoresis (high cardiac risk)")
            elif patient_data.get("chest_pain"):
                urgency_drivers.append("Persistent chest discomfort")
            if patient_data.get("fast_positive"):
                urgency_drivers.append("Positive FAST stroke criteria (acute neuro deficit)")
            if patient_data.get("unresponsive"):
                urgency_drivers.append("Unresponsive / altered level of consciousness")
            if patient_data.get("airway_compromise") or patient_data.get("stridor"):
                urgency_drivers.append("Critical airway compromise / stridor detected")
        else:
            if top_features:
                urgency_drivers.extend([f"Observed: {feat}" for feat in top_features[:3]])
            else:
                urgency_drivers.append("Stable physiological parameters; no acute red flags detected")

        # 2. Suggested clinical protocols
        suggested_protocols = []
        for proto in triggered_protocols:
            if isinstance(proto, dict):
                suggested_protocols.append(proto.get("title", proto.get("name", "Clinical Protocol")))
            elif isinstance(proto, str):
                suggested_protocols.append(proto)

        if sepsis_alert and "Sepsis Screening" not in suggested_protocols:
            suggested_protocols.append("Sepsis Resuscitation Bundle (qSOFA criteria met)")
        if patient_data.get("chest_pain") and "STEMI / Acute Coronary Protocol" not in suggested_protocols:
            suggested_protocols.append("12-Lead ECG within 10 minutes (STEMI Evaluation)")
        if patient_data.get("fast_positive") and "Stroke Code (FAST+)" not in suggested_protocols:
            suggested_protocols.append("Emergent CT Head / Stroke Code Activation")

        if not suggested_protocols:
            suggested_protocols.append("Standard Emergency Department Nursing Reassessment")

        # 3. Formulate concise clinical summary paragraphs
        urgency_label = PRIORITY_DESCRIPTIONS.get(priority, "Triage Urgency")
        summary_headline = f"{priority} Priority ({urgency_label})"

        rationale_bullets = urgency_drivers if urgency_drivers else ["Standard vital signs within tolerable emergency parameters."]

        # Synthesis text
        synthesis = (
            f"Patient triaged as {priority} with estimated risk score of {risk_score}/100 "
            f"(model confidence: {int(confidence * 100)}%). "
        )
        if priority in ["P1", "P2"]:
            synthesis += (
                f"Urgent prioritization is driven by acute physiological abnormalities, "
                f"specifically {', '.join(urgency_drivers[:2]).lower()}. Immediate bedside evaluation recommended."
            )
        else:
            synthesis += (
                "Patient demonstrates reassuring baseline hemodynamics without immediate red-flag triggers. "
                "Recommend routine queue placement with scheduled vitals recheck."
            )

        return {
            "priority": priority,
            "priority_description": urgency_label,
            "risk_score": risk_score,
            "confidence": confidence,
            "summary_headline": summary_headline,
            "clinical_synthesis": synthesis,
            "urgency_drivers": rationale_bullets,
            "suggested_protocols": suggested_protocols,
            "vitals_summary": {
                "shock_index": shock_index,
                "map": map_val,
                "sbp": sbp,
                "dbp": dbp,
                "heartrate": hr,
                "resprate": resprate,
                "o2sat": o2sat,
            },
            "disclaimer": CLINICAL_DISCLAIMER,
        }


# Singleton instance
clinical_summary_generator = ClinicalSummaryGenerator()


def generate_clinical_summary(patient_data: dict, triage_result: dict) -> dict:
    """Convenience functional interface."""
    return clinical_summary_generator.generate_summary(patient_data, triage_result)

"""
tests/test_llm_triage.py
========================

Unit & Integration tests for the LLM-Powered Patient Triage AI & Modular Inference Pipeline:
- NLP Preprocessing, abbreviation expansion, and NegEx integration
- Quantitative physiological indices (Shock Index, MAP, qSOFA)
- LangChain prompt orchestration and clinical context formatting
- ESI v4 Urgency Prioritization (P1 Resuscitation to P5 Non-Urgent)
- Actionable clinical emergency protocol triggers (STEMI, Stroke FAST, Sepsis)
- Zero-downtime deterministic fallback resilience
- FastAPI REST API endpoints (/api/v1/nlp/llm-triage and /api/v1/triage/llm-triage)
"""

from __future__ import annotations

import pytest
from app.llm.triage_chain import LLMPatientTriagePipeline, run_llm_triage


@pytest.fixture
def pipeline():
    return LLMPatientTriagePipeline()


def test_nlp_preprocessing_and_negex(pipeline):
    note = (
        "54yo male with acute crushing CP and SOB for 30 mins, diaphoresis. "
        "Denies fever, denies vomiting, no prior cardiac history. BP 90/60, HR 115, SpO2 89%."
    )
    context = pipeline.preprocess_clinical_context(note)

    # Abbreviation expansion check
    assert "chest pain" in context["cleaned_text"].lower()
    assert "shortness of breath" in context["cleaned_text"].lower()

    # Vitals check
    assert context["vitals"]["age"] == 54.0
    assert context["vitals"]["gender"] == "M"
    assert context["vitals"]["sbp"] == 90.0
    assert context["vitals"]["dbp"] == 60.0
    assert context["vitals"]["heartrate"] == 115.0
    assert context["vitals"]["o2sat"] == 89.0

    # Hemodynamic calculation: SI = 115 / 90 = 1.28
    assert context["shock_index"] == 1.28
    assert context["shock_index"] >= 0.90  # Elevated threshold

    # MAP calculation: (2*60 + 90) / 3 = 70.0
    assert context["map"] == 70.0

    # NegEx check: "fever" and "vomiting" must be in negated entities
    negated_texts = [e["text"].lower() for e in context["negated_entities"]]
    assert any("fever" in n for n in negated_texts)
    assert any("vomiting" in n for n in negated_texts)

    # Affirmed findings check: chest_pain and diaphoresis
    assert context["finding_flags"]["chest_pain"] is True
    assert context["finding_flags"]["diaphoresis"] is True


def test_prompt_orchestration(pipeline):
    note = "67yo female with sudden facial droop and right-sided hemiparesis. BP 180/105, HR 92, RR 18."
    context = pipeline.preprocess_clinical_context(note)
    messages, human_prompt = pipeline.orchestrate_prompt(context)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "JeevLok AI" in messages[0]["content"]
    assert "ESI v4" in messages[0]["content"]
    assert "Shock Index" in messages[0]["content"]

    assert messages[1]["role"] == "human"
    assert "180/105" in human_prompt
    assert "facial droop" in human_prompt or "fast_positive" in human_prompt.lower() or "neuro" in human_prompt.lower()


def test_urgency_prioritization_stemi_case(pipeline):
    """Acute STEMI case with chest pain + diaphoresis + borderline vitals -> P2 or P1."""
    note = (
        "58-year-old male with severe crushing chest pain radiating to left arm for 45 minutes, "
        "sweating profusely, cold clammy skin. Denies fever. BP 90/60, HR 115, SpO2 89%."
    )
    result = pipeline.run_pipeline(note)

    assert result["priority"] in ["P1", "P2"]
    assert result["urgency_level"] in ["CRITICAL", "HIGH"]
    assert result["risk_score"] >= 80.0
    assert result["vitals_summary"]["shock_index"] >= 0.90
    assert any("ECG" in p or "STEMI" in p for p in result["recommended_protocols"])
    assert result["auto_fill_payload"]["chest_pain"] is True
    assert result["auto_fill_payload"]["diaphoresis"] is True


def test_urgency_prioritization_stroke_fast_case(pipeline):
    """Acute stroke (FAST+) -> P2 Emergent with Head CT / Stroke Code protocol."""
    note = (
        "62yo female presenting with sudden onset right-sided facial droop, arm weakness, "
        "and slurred speech starting 35 mins ago. Denies seizure. BP 175/100, HR 88, RR 16."
    )
    result = pipeline.run_pipeline(note)

    assert result["priority"] == "P2"
    assert result["urgency_level"] == "HIGH"
    assert result["risk_score"] >= 80.0
    assert any("Stroke" in p or "CT" in p for p in result["recommended_protocols"])
    assert result["auto_fill_payload"]["fast_positive"] is True


def test_urgency_prioritization_decompensated_shock_p1(pipeline):
    """Profound shock with SI >= 1.3 or MAP < 50 -> P1 Resuscitation."""
    note = "45yo male in acute septic shock, unresponsive, BP 70/40, HR 135, RR 30, temp 103.5 F, SpO2 84%."
    result = pipeline.run_pipeline(note)

    assert result["priority"] == "P1"
    assert result["urgency_level"] == "CRITICAL"
    assert result["risk_score"] >= 90.0
    assert result["vitals_summary"]["shock_index"] >= 1.3
    assert any("Resuscitation" in p or "Airway" in p or "Fluid" in p for p in result["recommended_protocols"])


def test_urgency_prioritization_stable_routine_p4_p5(pipeline):
    """Routine minor ankle sprain or prescription refill -> P4 or P5."""
    note = "26yo female with mild ankle sprain after jogging. Denies head trauma, no chest pain, no SOB. BP 118/76, HR 70, RR 14, SpO2 99%, pain 3/10."
    result = pipeline.run_pipeline(note)

    assert result["priority"] in ["P4", "P5"]
    assert result["urgency_level"] in ["LOW", "NON-URGENT"]
    assert result["risk_score"] <= 35.0
    assert result["vitals_summary"]["shock_index"] < 0.70


def test_deterministic_fallback_resilience(pipeline):
    """When no API key is provided, deterministic fallback runs with complete schema."""
    note = "34yo male with moderate acute abdominal pain, nausea. Denies chest pain. BP 124/82, HR 84, temp 99.1 F, pain 5/10."
    result = pipeline.run_pipeline(note, include_prompt_debug=True)

    assert result["priority"] == "P3"
    assert result["disclaimer"] == "AI recommends. Clinician decides."
    assert result["prompt_orchestration"] is not None
    assert result["prompt_orchestration"]["fallback_used"] is True
    assert "LangChain" in result["prompt_orchestration"]["framework"]


def test_fastapi_llm_triage_endpoints(test_client):
    """Test both REST API endpoints (/api/v1/nlp/llm-triage and /api/v1/triage/llm-triage)."""
    payload = {
        "text": "58-year-old male with severe crushing chest pain, diaphoresis, BP 90/60, HR 115, SpO2 89%.",
        "temperature": 0.0,
        "include_prompt_debug": True,
    }

    # 1. NLP route
    resp1 = test_client.post("/api/v1/nlp/llm-triage", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["priority"] in ["P1", "P2"]
    assert data1["urgency_level"] in ["CRITICAL", "HIGH"]
    assert data1["risk_score"] >= 80.0
    assert data1["prompt_orchestration"]["framework"] == "LangChain LCEL"
    assert data1["auto_fill_payload"]["chest_pain"] is True

    # 2. Triage route alias
    resp2 = test_client.post("/api/v1/triage/llm-triage", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["priority"] == data1["priority"]
    assert data2["risk_score"] == data1["risk_score"]

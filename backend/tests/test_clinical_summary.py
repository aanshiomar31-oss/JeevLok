"""
tests/test_clinical_summary.py
==============================

Unit tests for LLM Clinical Summary generation:
- Urgency driver synthesis
- Physiological index calculations (Shock Index, MAP)
- Suggested clinical protocol activations
- Mandatory safety disclaimer enforcement
"""

import pytest
from app.llm.clinical_summary import ClinicalSummaryGenerator, CLINICAL_DISCLAIMER


def test_clinical_summary_generation_p2_cardiac():
    generator = ClinicalSummaryGenerator()
    patient_data = {
        "age": 58,
        "gender": "M",
        "sbp": 88,
        "dbp": 54,
        "heartrate": 118,
        "resprate": 24,
        "o2sat": 91,
        "chest_pain": True,
        "diaphoresis": True,
    }
    triage_result = {
        "priority": "P2",
        "risk_score": 82,
        "confidence": 0.88,
        "top_features": ["Shock Index: 1.34", "Systolic BP: 88", "Chest Pain Present"],
        "triggered_protocols": [{"code": "STEMI", "title": "STEMI / Acute Coronary Protocol"}],
        "sepsis_alert": False,
    }

    summary = generator.generate_summary(patient_data, triage_result)

    assert summary["priority"] == "P2"
    assert summary["risk_score"] == 82
    assert summary["disclaimer"] == CLINICAL_DISCLAIMER
    assert summary["vitals_summary"]["shock_index"] == round(118 / 88, 2)
    assert any("Shock Index" in driver for driver in summary["urgency_drivers"])
    assert any("STEMI" in proto for proto in summary["suggested_protocols"])


def test_clinical_summary_generation_stable_p4():
    generator = ClinicalSummaryGenerator()
    patient_data = {
        "age": 25,
        "gender": "F",
        "sbp": 120,
        "dbp": 80,
        "heartrate": 72,
        "resprate": 16,
        "o2sat": 99,
        "chest_pain": False,
        "diaphoresis": False,
    }
    triage_result = {
        "priority": "P4",
        "risk_score": 15,
        "confidence": 0.94,
        "top_features": ["Normal Heart Rate", "Normal Blood Pressure"],
        "triggered_protocols": [],
        "sepsis_alert": False,
    }

    summary = generator.generate_summary(patient_data, triage_result)

    assert summary["priority"] == "P4"
    assert summary["risk_score"] == 15
    assert summary["disclaimer"] == CLINICAL_DISCLAIMER
    assert summary["vitals_summary"]["shock_index"] == round(72 / 120, 2)

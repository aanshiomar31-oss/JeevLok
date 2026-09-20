"""
tests/test_chat_copilot.py
==========================

Unit tests for Intelligent Chat Assistant (Clinical Copilot):
- Document retrieval from knowledge base
- Shock index and MAP explanations
- Patient context incorporation
- Safety disclaimer presence
"""

import pytest
from app.llm.rag_engine import ClinicalRAGEngine, CLINICAL_DISCLAIMER


def test_copilot_retrieval_shock_index():
    engine = ClinicalRAGEngine()
    docs = engine.retrieve_relevant_docs("Why is shock index high?")
    assert len(docs) > 0
    assert any("Shock Index" in d["title"] for d in docs)


def test_copilot_answer_shock_index_with_patient_context():
    engine = ClinicalRAGEngine()
    patient_context = {
        "heartrate": 120,
        "sbp": 90,
        "priority": "P2",
        "risk_score": 75,
    }
    res = engine.answer_query(
        query="Why is shock index high?",
        session_id="test_session_1",
        patient_context=patient_context,
    )

    assert "reply" in res
    assert "Shock Index" in res["reply"]
    assert "1.33" in res["reply"] or "1.3" in res["reply"]
    assert res["disclaimer"] == CLINICAL_DISCLAIMER


def test_copilot_answer_map_query():
    engine = ClinicalRAGEngine()
    patient_context = {
        "sbp": 90,
        "dbp": 60,
    }
    res = engine.answer_query(
        query="What does MAP mean?",
        session_id="test_session_2",
        patient_context=patient_context,
    )

    assert "Mean Arterial Pressure" in res["reply"]
    assert "70.0" in res["reply"] or "MAP" in res["reply"]
    assert res["disclaimer"] == CLINICAL_DISCLAIMER

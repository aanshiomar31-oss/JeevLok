"""
tests/test_nlp_pipeline.py
==========================

Unit tests for the Clinical NLP Subsystem:
- Preprocessing and abbreviation expansion
- NegEx clinical negation detection
- Named entity recognition and vitals extraction
- Semantic concept normalization and ICD-10 mapping
- End-to-end triage form auto-fill payload generation
"""

import pytest
from app.nlp.preprocessor import ClinicalPreprocessor
from app.nlp.negation import ClinicalNegationDetector
from app.nlp.entity_extractor import ClinicalEntityExtractor
from app.nlp.similarity import ClinicalSemanticMatcher
from app.nlp.pipeline import ClinicalNLPPipeline


def test_clinical_preprocessor_abbreviations():
    preprocessor = ClinicalPreprocessor()
    text = "54yo pt with acute SOB and CP, hx of HTN"
    cleaned = preprocessor.clean_text(text)

    assert "shortness of breath" in cleaned.lower()
    assert "chest pain" in cleaned.lower()
    assert "patient" in cleaned.lower()
    assert "hypertension" in cleaned.lower()


def test_negation_detection_affirmed_vs_negated():
    detector = ClinicalNegationDetector()
    text = "Patient reports severe chest pain but denies fever or vomiting."

    # "chest pain" at index ~15
    cp_start = text.find("chest pain")
    cp_end = cp_start + len("chest pain")
    is_neg, trigger = detector.is_negated(text, (cp_start, cp_end))
    assert is_neg is False

    # "fever" at index ~44
    fever_start = text.find("fever")
    fever_end = fever_start + len("fever")
    is_neg_fever, trigger_fever = detector.is_negated(text, (fever_start, fever_end))
    assert is_neg_fever is True
    assert "denies" in trigger_fever.lower()

    # "vomiting" at index ~53
    vomit_start = text.find("vomiting")
    vomit_end = vomit_start + len("vomiting")
    is_neg_vomit, _ = detector.is_negated(text, (vomit_start, vomit_end))
    assert is_neg_vomit is True


def test_vitals_and_demographics_extraction():
    extractor = ClinicalEntityExtractor()
    note = "62yo female presenting with BP 85/55, HR 122, RR 26, temp 101.5 F, O2 89%, pain 8/10"
    vitals = extractor.extract_vitals_and_demographics(note)

    assert vitals["age"] == 62.0
    assert vitals["gender"] == "F"
    assert vitals["sbp"] == 85.0
    assert vitals["dbp"] == 55.0
    assert vitals["heartrate"] == 122.0
    assert vitals["resprate"] == 26.0
    assert vitals["temperature"] == 101.5
    assert vitals["o2sat"] == 89.0
    assert vitals["pain"] == 8.0


def test_semantic_matching_and_icd10():
    matcher = ClinicalSemanticMatcher()

    # Exact synonym
    match1 = matcher.match_concept("crushing chest pain")
    assert match1 is not None
    assert match1["category"] == "chest_pain"
    assert match1["icd10"] == "R07.9"

    # Colloquial synonym
    match2 = matcher.match_concept("heart attack feeling")
    assert match2 is not None
    assert match2["category"] == "chest_pain"
    assert match2["icd10"] == "R07.9"

    # Dyspnea
    match3 = matcher.match_concept("shortness of breath")
    assert match3 is not None
    assert match3["category"] == "dyspnea"
    assert match3["icd10"] == "R06.02"


def test_end_to_end_nlp_pipeline():
    pipeline = ClinicalNLPPipeline()
    note = (
        "58-year-old male with crushing chest pain for 45 minutes, sweating profusely "
        "and difficulty breathing. Denies fever or trauma. BP 90/60, HR 115, O2 88%."
    )
    result = pipeline.process_clinical_note(note)

    # Vitals check
    assert result["vitals"]["age"] == 58.0
    assert result["vitals"]["gender"] == "M"
    assert result["vitals"]["sbp"] == 90.0
    assert result["vitals"]["dbp"] == 60.0
    assert result["vitals"]["heartrate"] == 115.0
    assert result["vitals"]["o2sat"] == 88.0

    # Findings check (affirmed vs negated)
    assert result["findings"]["chest_pain"] is True
    assert result["findings"]["diaphoresis"] is True
    assert "fever" in result["negated_findings"]
    assert "trauma" in result["negated_findings"]

    # Urgency check
    assert result["urgency_level"] in ["CRITICAL", "HIGH"]
    assert any("crushing" in kw.lower() for kw in result["urgency_keywords"])

    # Auto-fill payload verification
    payload = result["auto_fill_payload"]
    assert payload["chest_pain"] is True
    assert payload["diaphoresis"] is True
    assert payload["sbp"] == 90.0
    assert payload["heartrate"] == 115.0

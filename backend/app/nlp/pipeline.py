"""
app/nlp/pipeline.py
===================

Unified Clinical NLP Pipeline for Emergency Department Triage.
Orchestrates:
1. Text cleaning, tokenization, abbreviation expansion
2. NegEx-style clinical negation detection
3. Medical entity recognition (findings, vitals, demographics)
4. Concept normalization & ICD-10 mapping
5. Structured feature generation for auto-filling the triage form
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

from app.nlp.preprocessor import ClinicalPreprocessor
from app.nlp.negation import ClinicalNegationDetector
from app.nlp.entity_extractor import ClinicalEntityExtractor
from app.nlp.similarity import ClinicalSemanticMatcher


class ClinicalNLPPipeline:
    """
    End-to-end Clinical NLP pipeline that parses unstructured triage notes
    into structured clinical signals for machine learning inference and form auto-fill.
    """

    def __init__(self) -> None:
        self.preprocessor = ClinicalPreprocessor()
        self.negation_detector = ClinicalNegationDetector()
        self.entity_extractor = ClinicalEntityExtractor()
        self.semantic_matcher = ClinicalSemanticMatcher()

    def process_clinical_note(self, text: str) -> Dict[str, Any]:
        """
        Process free-text clinical symptoms and return structured triage features.

        Example input:
            "54yo male with crushing chest pain for 30 minutes, sweating profusely and difficulty breathing.
             Denies fever or vomiting. BP 90/60, HR 115, O2 89%."

        Returns structured JSON with extracted vitals, finding flags, negations,
        urgency classification, ICD-10 codes, and auto-fill payload.
        """
        if not text or not text.strip():
            return self._empty_result()

        # Step 1: Preprocess text (clean & expand abbreviations)
        preprocessed = self.preprocessor.preprocess(text)
        cleaned_text = preprocessed["cleaned_text"]

        # Step 2: Extract raw clinical entities (mentions)
        raw_entities = self.entity_extractor.extract_raw_entities(cleaned_text)

        # Step 3: Run Negation Detection on extracted entities
        affirmed_entities, negated_entities = self.negation_detector.filter_negated_entities(
            cleaned_text, raw_entities
        )

        # Step 4: Extract numerical vitals and demographics
        vitals = self.entity_extractor.extract_vitals_and_demographics(cleaned_text)

        # Step 5: Detect urgency level and urgency keywords
        urgency_level, urgency_keywords = self.entity_extractor.detect_urgency_level(cleaned_text)

        # Step 6: Map affirmed entities to ICD-10 concepts
        symptom_phrases = [e["text"] for e in affirmed_entities]
        icd_mappings = self.semantic_matcher.map_symptoms_to_icd10(symptom_phrases)

        # Step 7: Build finding boolean flags (only affirmed findings are True!)
        findings_flags = {
            "chest_pain": False,
            "diaphoresis": False,
            "fast_positive": False,
            "unresponsive": False,
            "seizing": False,
            "airway_compromise": False,
            "stridor": False,
        }
        for entity in affirmed_entities:
            cat = entity["category"]
            if cat in findings_flags:
                findings_flags[cat] = True

        negated_finding_names = [e["category"] for e in negated_entities]

        # Step 8: Build Auto-Fill Form Payload
        auto_fill_payload = {
            "age": vitals.get("age"),
            "gender": vitals.get("gender"),
            "heartrate": vitals.get("heartrate"),
            "sbp": vitals.get("sbp"),
            "dbp": vitals.get("dbp"),
            "resprate": vitals.get("resprate"),
            "temperature": vitals.get("temperature"),
            "o2sat": vitals.get("o2sat"),
            "pain": vitals.get("pain"),
            "chief_complaint": self._derive_chief_complaint(affirmed_entities, cleaned_text),
            "chest_pain": findings_flags["chest_pain"],
            "diaphoresis": findings_flags["diaphoresis"],
            "fast_positive": findings_flags["fast_positive"],
            "unresponsive": findings_flags["unresponsive"],
            "seizing": findings_flags["seizing"],
            "airway_compromise": findings_flags["airway_compromise"],
            "stridor": findings_flags["stridor"],
        }

        # Step 9: Assemble all entity chips for UI presentation
        entity_chips = []
        for e in affirmed_entities:
            entity_chips.append({
                "label": e["text"],
                "category": e["category"],
                "is_negated": False,
                "confidence": e.get("confidence", 0.95),
                "type": "finding",
            })
        for e in negated_entities:
            entity_chips.append({
                "label": f"No {e['text']}",
                "category": e["category"],
                "is_negated": True,
                "confidence": e.get("confidence", 0.95),
                "type": "negation",
            })
        for k, v in vitals.items():
            if v is not None:
                entity_chips.append({
                    "label": f"{k.upper()}: {v}",
                    "category": k,
                    "is_negated": False,
                    "confidence": 1.0,
                    "type": "vital",
                })

        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "urgency_level": urgency_level,
            "urgency_keywords": urgency_keywords,
            "vitals": vitals,
            "findings": findings_flags,
            "negated_findings": negated_finding_names,
            "affirmed_entities": affirmed_entities,
            "negated_entities": negated_entities,
            "entity_chips": entity_chips,
            "icd_mappings": icd_mappings,
            "auto_fill_payload": auto_fill_payload,
        }

    def _derive_chief_complaint(self, affirmed_entities: List[dict], text: str) -> str:
        """Derive a concise chief complaint string from affirmed findings."""
        if affirmed_entities:
            unique_terms = list(dict.fromkeys(e["text"].capitalize() for e in affirmed_entities))
            return ", ".join(unique_terms[:3])

        # Fallback to first 40 chars of text
        snippet = text[:40].strip()
        return snippet if snippet else "Acute Presentation"

    def _empty_result(self) -> Dict[str, Any]:
        """Return a clean empty result schema."""
        return {
            "raw_text": "",
            "cleaned_text": "",
            "urgency_level": "LOW",
            "urgency_keywords": [],
            "vitals": {},
            "findings": {
                "chest_pain": False,
                "diaphoresis": False,
                "fast_positive": False,
                "unresponsive": False,
                "seizing": False,
                "airway_compromise": False,
                "stridor": False,
            },
            "negated_findings": [],
            "affirmed_entities": [],
            "negated_entities": [],
            "entity_chips": [],
            "icd_mappings": [],
            "auto_fill_payload": {
                "age": None,
                "gender": None,
                "heartrate": None,
                "sbp": None,
                "dbp": None,
                "resprate": None,
                "temperature": None,
                "o2sat": None,
                "pain": None,
                "chief_complaint": "",
                "chest_pain": False,
                "diaphoresis": False,
                "fast_positive": False,
                "unresponsive": False,
                "seizing": False,
                "airway_compromise": False,
                "stridor": False,
            },
        }


# Global singleton instance for high performance
nlp_pipeline = ClinicalNLPPipeline()


def parse_clinical_text(text: str) -> Dict[str, Any]:
    """Helper functional interface."""
    return nlp_pipeline.process_clinical_note(text)

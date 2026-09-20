"""
app/nlp/preprocessor.py
=======================

Clinical NLP Preprocessor for Emergency Department Text.
Provides text normalization, abbreviation expansion, clinical tokenization,
and lemmatization.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Common emergency department clinical abbreviations and colloquialisms
CLINICAL_ABBREVIATIONS = {
    r"\bsob\b": "shortness of breath",
    r"\bcp\b": "chest pain",
    r"\bha\b": "headache",
    r"\bn/v\b": "nausea and vomiting",
    r"\bnv\b": "nausea and vomiting",
    r"\bloc\b": "loss of consciousness",
    r"\bpt\b": "patient",
    r"\bhx\b": "history",
    r"\bdn\b": "denies",
    r"\bw/\b": "with",
    r"\bw/o\b": "without",
    r"\byo\b": "year old",
    r"\by/o\b": "year old",
    r"\byr\b": "year old",
    r"\byrs\b": "years",
    r"\bfx\b": "fracture",
    r"\btx\b": "treatment",
    r"\bsx\b": "symptoms",
    r"\bdx\b": "diagnosis",
    r"\brx\b": "prescription",
    r"\bhtn\b": "hypertension",
    r"\bdm\b": "diabetes mellitus",
    r"\bcad\b": "coronary artery disease",
    r"\bchf\b": "congestive heart failure",
    r"\bcva\b": "cerebrovascular accident stroke",
    r"\btia\b": "transient ischemic attack",
    r"\bami\b": "acute myocardial infarction",
    r"\bmi\b": "myocardial infarction",
    r"\bstemi\b": "st elevation myocardial infarction",
    r"\bpe\b": "pulmonary embolism",
    r"\bdvt\b": "deep vein thrombosis",
    r"\bmva\b": "motor vehicle accident",
    r"\bmvc\b": "motor vehicle collision",
}

# Common clinical suffix lemmatization rules (fallback without heavy NLP libraries)
LEMMA_SUFFIX_RULES = [
    (r"ing$", ""),
    (r"ed$", ""),
    (r"es$", ""),
    (r"s$", ""),
]


class ClinicalPreprocessor:
    """Preprocesses raw clinical triage text for downstream NER and classification."""

    def __init__(self) -> None:
        # Pre-compile abbreviation regexes
        self.abbreviation_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in CLINICAL_ABBREVIATIONS.items()
        ]

    def expand_abbreviations(self, text: str) -> str:
        """Expand clinical abbreviations (e.g., 'SOB' -> 'shortness of breath')."""
        result = text
        for pattern, replacement in self.abbreviation_patterns:
            result = pattern.sub(replacement, result)
        return result

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace, handle medical punctuation, expand abbreviations,
        and clean extraneous symbols while preserving numbers, units, and slashes
        (e.g., BP 120/80, 98.6 F).
        """
        if not text:
            return ""

        # Normalize unicode whitespace and quotes
        text = re.sub(r"[\u2018\u2019]", "'", text)
        text = re.sub(r"[\u201C\u201D]", '"', text)

        # Expand abbreviations
        text = self.expand_abbreviations(text)

        # Collapse excess whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize clinical text into word tokens, preserving compound terms like
        'year-old' and vital signs like '120/80'.
        """
        cleaned = self.clean_text(text)
        # Tokenize by words and clinically relevant tokens (e.g., 120/80, 10/10)
        tokens = re.findall(r"\b\d+/\d+\b|\b\d+\.?\d*\b|\b[\w'-]+\b", cleaned.lower())
        return tokens

    def lemmatize_token(self, token: str) -> str:
        """Lightweight clinical lemmatization."""
        # Don't touch numbers or acronyms
        if re.search(r"\d", token) or len(token) <= 3:
            return token

        # Special clinical lemmas
        special_cases = {
            "vomiting": "vomit",
            "sweating": "sweat",
            "wheezing": "wheeze",
            "gasping": "gasp",
            "seizing": "seizure",
            "convulsing": "convulsion",
            "bleeding": "bleed",
            "breathing": "breathe",
            "dizzy": "dizziness",
        }
        if token in special_cases:
            return special_cases[token]

        for suffix, replacement in LEMMA_SUFFIX_RULES:
            if re.search(suffix, token) and len(token) > len(suffix) + 3:
                return re.sub(suffix, replacement, token)

        return token

    def preprocess(self, text: str) -> dict:
        """Complete preprocessing pipeline output."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        lemmas = [self.lemmatize_token(t) for t in tokens]

        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "lemmas": lemmas,
        }

"""
app/nlp
=======

Clinical NLP and Medical Entity Recognition Subsystem.
"""

from app.nlp.pipeline import ClinicalNLPPipeline, nlp_pipeline, parse_clinical_text
from app.nlp.preprocessor import ClinicalPreprocessor
from app.nlp.negation import ClinicalNegationDetector
from app.nlp.entity_extractor import ClinicalEntityExtractor
from app.nlp.similarity import ClinicalSemanticMatcher

__all__ = [
    "ClinicalNLPPipeline",
    "nlp_pipeline",
    "parse_clinical_text",
    "ClinicalPreprocessor",
    "ClinicalNegationDetector",
    "ClinicalEntityExtractor",
    "ClinicalSemanticMatcher",
]

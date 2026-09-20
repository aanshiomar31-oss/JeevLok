"""
app/nlp/negation.py
===================

Clinical Negation Detection Module (NegEx-inspired).
Determines whether a clinical entity or symptom mentioned in clinical text
is affirmed or negated (e.g., "denies chest pain", "no fever", "without SOB").
"""

from __future__ import annotations

import re
from typing import List, Tuple, Set


# Pre-negation triggers (negate terms that appear AFTER the trigger)
PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdenying\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\brules out\b",
    r"\bruled out\b",
    r"\brule out\b",
    r"\bno signs of\b",
    r"\bno evidence of\b",
    r"\bfree of\b",
    r"\bnever had\b",
    r"\bnot experiencing\b",
    r"\babsence of\b",
    r"\babsent\b",
    r"\bnon-febrile\b",
    r"\bafebrile\b",
]

# Post-negation triggers (negate terms that appear BEFORE the trigger)
POST_NEGATION_TRIGGERS = [
    r"\bwas ruled out\b",
    r"\bis ruled out\b",
    r"\bunlikely\b",
    r"\bis absent\b",
    r"\bwas absent\b",
    r"\bnot present\b",
    r"\bresolved\b",
]

# Pseudo-negation phrases (phrases containing negation words that do NOT negate clinical findings)
PSEUDO_NEGATIONS = [
    r"\bno change\b",
    r"\bno better\b",
    r"\bno worse\b",
    r"\bnot only\b",
    r"\bwithout difficulty\b",
]

# Conjunctions and punctuation that terminate the scope of a negation
SCOPE_TERMINATORS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\bexcept\b",
    r"\baside from\b",
    r"\bnevertheless\b",
    r"\byet\b",
    r";",
    r"\.",
]


class ClinicalNegationDetector:
    """
    Identifies if a given clinical finding or symptom is negated in free text.
    Uses regex boundary matching, pseudo-negation filtering, and directional scope windows.
    """

    def __init__(self, window_size: int = 6) -> None:
        self.window_size = window_size
        self.pre_patterns = [re.compile(p, re.IGNORECASE) for p in PRE_NEGATION_TRIGGERS]
        self.post_patterns = [re.compile(p, re.IGNORECASE) for p in POST_NEGATION_TRIGGERS]
        self.pseudo_patterns = [re.compile(p, re.IGNORECASE) for p in PSEUDO_NEGATIONS]
        self.terminator_patterns = [re.compile(p, re.IGNORECASE) for p in SCOPE_TERMINATORS]

    def is_negated(self, text: str, entity_span: Tuple[int, int]) -> Tuple[bool, str | None]:
        """
        Check if the entity located at `entity_span` (start_char, end_char) in `text`
        is negated.

        Returns:
            (is_negated: bool, trigger_found: str | None)
        """
        start_idx, end_idx = entity_span

        # 1. Check for pseudo-negations that might span across the entity
        for pseudo_pat in self.pseudo_patterns:
            for match in pseudo_pat.finditer(text):
                if match.start() <= start_idx and match.end() >= end_idx:
                    return False, None

        # 2. Inspect the pre-entity context window (up to ~60 characters or preceding words)
        pre_context = text[max(0, start_idx - 60):start_idx]

        # Truncate pre-context at the rightmost scope terminator if one exists
        for term_pat in self.terminator_patterns:
            matches = list(term_pat.finditer(pre_context))
            if matches:
                # Keep only context after the last terminator
                last_match = matches[-1]
                pre_context = pre_context[last_match.end():]

        # Check for pre-negation triggers
        for pre_pat in self.pre_patterns:
            match = pre_pat.search(pre_context)
            if match:
                return True, match.group(0)

        # 3. Inspect the post-entity context window (up to ~40 characters following the entity)
        post_context = text[end_idx:min(len(text), end_idx + 40)]

        # Truncate post-context at the leftmost scope terminator if one exists
        for term_pat in self.terminator_patterns:
            match = term_pat.search(post_context)
            if match:
                post_context = post_context[:match.start()]

        # Check for post-negation triggers
        for post_pat in self.post_patterns:
            match = post_pat.search(post_context)
            if match:
                return True, match.group(0)

        return False, None

    def filter_negated_entities(self, text: str, entities: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Partition a list of extracted entities into affirmed and negated entities.
        Each entity is expected to have 'start' and 'end' character offsets.
        """
        affirmed = []
        negated = []

        for entity in entities:
            is_neg, trigger = self.is_negated(text, (entity["start"], entity["end"]))
            entity_copy = dict(entity)
            entity_copy["is_negated"] = is_neg
            entity_copy["negation_trigger"] = trigger

            if is_neg:
                negated.append(entity_copy)
            else:
                affirmed.append(entity_copy)

        return affirmed, negated

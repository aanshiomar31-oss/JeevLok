"""
app/nlp/entity_extractor.py
===========================

Clinical Named Entity Recognition (NER) & Feature Extractor.
Extracts symptoms, clinical finding flags, numerical vitals, demographics,
and urgency markers from free-text emergency department notes.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any

# Clinical finding concepts mapped to search patterns and normalized names
CLINICAL_FINDINGS_PATTERNS = {
    "chest_pain": [
        r"\bchest\s+pain\b",
        r"\bcrushing\s+chest\s+pain\b",
        r"\bchest\s+discomfort\b",
        r"\bchest\s+pressure\b",
        r"\btightness\s+in\s+chest\b",
        r"\bchest\s+tightness\b",
        r"\bangina\b",
        r"\bsubsternal\s+chest\s+pain\b",
        r"\bheart\s+attack\s+feeling\b",
        r"\bheaviness\s+in\s+chest\b",
    ],
    "diaphoresis": [
        r"\bdiaphoresis\b",
        r"\bdiaphoretic\b",
        r"\bsweating\b",
        r"\bprofuse\s+sweating\b",
        r"\bcold\s+sweats?\b",
        r"\bclammy\s+skin\b",
        r"\bdrenched\s+in\s+sweat\b",
        r"\bexcessive\s+perspiration\b",
    ],
    "dyspnea": [
        r"\bdyspnea\b",
        r"\bshortness\s+of\s+breath\b",
        r"\bsob\b",
        r"\bdifficulty\s+breathing\b",
        r"\bbreathlessness\b",
        r"\bwheezing\b",
        r"\bgasping\b",
        r"\bair\s+hunger\b",
        r"\brespiratory\s+distress\b",
        r"\blabored\s+breathing\b",
    ],
    "fast_positive": [
        r"\bfast[- ]positive\b",
        r"\bfacial\s+droop\b",
        r"\bface\s+droop\b",
        r"\barm\s+weakness\b",
        r"\bslurred\s+speech\b",
        r"\bspeech\s+difficulty\b",
        r"\bsudden\s+numbness\b",
        r"\bunilateral\s+weakness\b",
        r"\bhemiparesis\b",
        r"\bstroke\s+symptoms?\b",
        r"\bsigns\s+of\s+stroke\b",
        r"\bvision\s+loss\b",
    ],
    "unresponsive": [
        r"\bunresponsive\b",
        r"\bunconscious\b",
        r"\bloss\s+of\s+consciousness\b",
        r"\bloc\b",
        r"\bpassed\s+out\b",
        r"\bsyncope\b",
        r"\bblacked\s+out\b",
        r"\bcomatose\b",
        r"\blethargic\b",
        r"\baltered\s+mental\s+status\b",
    ],
    "seizing": [
        r"\bseizing\b",
        r"\bseizure\b",
        r"\bconvulsions?\b",
        r"\btonic[- ]clonic\b",
        r"\bstatus\s+epilepticus\b",
        r"\bfitting\b",
    ],
    "airway_compromise": [
        r"\bairway\s+compromise\b",
        r"\bairway\s+obstruction\b",
        r"\bchoking\b",
        r"\btongue\s+swelling\b",
        r"\bcannot\s+protect\s+airway\b",
        r"\bforeign\s+body\s+in\s+airway\b",
    ],
    "stridor": [
        r"\bstridor\b",
        r"\bstridorous\b",
        r"\bhigh[- ]pitched\s+breathing\b",
        r"\bharsh\s+inspiratory\s+sound\b",
    ],
    "vomiting": [
        r"\bvomiting\b",
        r"\bvomit\b",
        r"\bemesis\b",
        r"\bthrowing\s+up\b",
        r"\bthrew\s+up\b",
        r"\bnausea\b",
        r"\bnauseated\b",
        r"\bn/v\b",
    ],
    "fever": [
        r"\bfever\b",
        r"\bfebrile\b",
        r"\bchills\b",
        r"\bhigh\s+temperature\b",
        r"\bhot\s+to\s+touch\b",
        r"\bpyrexia\b",
        r"\bburning\s+up\b",
    ],
    "trauma": [
        r"\btrauma\b",
        r"\bcar\s+crash\b",
        r"\bmotor\s+vehicle\s+accident\b",
        r"\bmva\b",
        r"\bmvc\b",
        r"\bfall\b",
        r"\bfell\b",
        r"\bblunt\s+force\b",
        r"\blaceration\b",
        r"\bhead\s+injury\b",
        r"\bfracture\b",
    ],
    "bleeding": [
        r"\bbleeding\b",
        r"\bhemorrhage\b",
        r"\bblood\s+loss\b",
        r"\bhematemesis\b",
        r"\brectal\s+bleeding\b",
        r"\bmassive\s+bleed\b",
        r"\bwound\s+bleeding\b",
    ],
}

# Urgency keywords and modifiers
URGENCY_KEYWORDS = {
    "CRITICAL": [
        r"\bcrushing\b",
        r"\bsevere\b",
        r"\bunbearable\b",
        r"\bworst\s+headache\s+of\s+life\b",
        r"\bthunderclap\b",
        r"\bcyanosis\b",
        r"\bcyanotic\b",
        r"\bunconscious\b",
        r"\bmassive\s+bleed\b",
        r"\bradiating\s+to\s+left\s+arm\b",
        r"\bradiating\s+to\s+jaw\b",
        r"\bshock\b",
        r"\bcardiac\s+arrest\b",
    ],
    "HIGH": [
        r"\bworsening\b",
        r"\bacute\b",
        r"\bpersistent\b",
        r"\bintense\b",
        r"\bheavy\b",
        r"\bsharp\b",
        r"\bfeverish\b",
    ],
    "MODERATE": [
        r"\bmoderate\b",
        r"\bdull\b",
        r"\bon and off\b",
        r"\bintermittent\b",
        r"\brecurrent\b",
    ],
    "LOW": [
        r"\bmild\b",
        r"\bslight\b",
        r"\boccasional\b",
        r"\bminor\b",
        r"\bchronic\b",
    ],
}


class ClinicalEntityExtractor:
    """Extracts clinical concepts, findings, vitals, and demographics from text."""

    def __init__(self) -> None:
        # Compile clinical finding patterns
        self.finding_regexes = {}
        for category, patterns in CLINICAL_FINDINGS_PATTERNS.items():
            self.finding_regexes[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Compile urgency keyword patterns
        self.urgency_regexes = {
            level: [re.compile(p, re.IGNORECASE) for p in patterns]
            for level, patterns in URGENCY_KEYWORDS.items()
        }

    def extract_raw_entities(self, text: str) -> List[Dict[str, Any]]:
        """Find all matching clinical finding mentions in text with their spans."""
        entities = []
        for category, regex_list in self.finding_regexes.items():
            for regex in regex_list:
                for match in regex.finditer(text):
                    entities.append({
                        "category": category,
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.95,
                    })

        # Sort entities by start position and deduplicate overlapping spans
        entities.sort(key=lambda e: e["start"])
        deduped = []
        for e in entities:
            if not deduped:
                deduped.append(e)
            else:
                last = deduped[-1]
                # If overlap, keep the longer span
                if e["start"] < last["end"]:
                    if (e["end"] - e["start"]) > (last["end"] - last["start"]):
                        deduped[-1] = e
                else:
                    deduped.append(e)

        return deduped

    def extract_vitals_and_demographics(self, text: str) -> Dict[str, Any]:
        """
        Extract numerical vitals and demographics from free text using robust regex.
        Supports standard clinical notation like:
        - "BP 120/80", "BP: 90/60", "blood pressure 135 over 85"
        - "HR 115", "heart rate 120 bpm", "pulse 98"
        - "RR 24", "resp rate 22", "breathing 28/min"
        - "temp 102.4", "temperature 38.5 C", "99.1 F"
        - "O2 88%", "spo2 94%", "o2 sat 92%"
        - "pain 9/10", "pain 8 out of 10"
        - "54yo male", "62 year old woman", "age 45"
        """
        vitals: Dict[str, Any] = {
            "age": None,
            "gender": None,
            "sbp": None,
            "dbp": None,
            "heartrate": None,
            "resprate": None,
            "temperature": None,
            "o2sat": None,
            "pain": None,
        }

        # 1. Blood Pressure: e.g. "BP 120/80", "120/80 mmHg", "120 over 80"
        bp_match = re.search(
            r"(?:bp|blood\s+pressure)?\s*:?\s*(\b\d{2,3})\s*(?:/|\bover\b)\s*(\d{2,3})\s*(?:mmhg)?\b",
            text,
            re.IGNORECASE,
        )
        if bp_match:
            sbp = float(bp_match.group(1))
            dbp = float(bp_match.group(2))
            if 50 <= sbp <= 260 and 30 <= dbp <= 160:
                vitals["sbp"] = sbp
                vitals["dbp"] = dbp

        # 2. Heart Rate / Pulse: e.g. "HR 115", "pulse 102", "heart rate: 88"
        hr_match = re.search(
            r"\b(?:hr|pulse|heart\s+rate|pulse\s+rate)\s*:?\s*(\d{2,3})\s*(?:bpm)?\b",
            text,
            re.IGNORECASE,
        )
        if hr_match:
            hr = float(hr_match.group(1))
            if 30 <= hr <= 250:
                vitals["heartrate"] = hr

        # 3. Respiratory Rate: e.g. "RR 24", "resp rate 22", "respirations 28"
        rr_match = re.search(
            r"\b(?:rr|resp\s+rate|respiratory\s+rate|respirations|breathing)\s*:?\s*(\d{1,2})\s*(?:/min|bpm)?\b",
            text,
            re.IGNORECASE,
        )
        if rr_match:
            rr = float(rr_match.group(1))
            if 6 <= rr <= 70:
                vitals["resprate"] = rr

        # 4. Temperature: e.g. "temp 102.4", "temperature 38.5 C", "99.2 F"
        temp_match = re.search(
            r"\b(?:temp|temperature)\s*:?\s*(\d{2,3}\.?\d?)\s*(?:[°º]?\s*[cf])?\b",
            text,
            re.IGNORECASE,
        )
        if temp_match:
            temp = float(temp_match.group(1))
            if 30.0 <= temp <= 110.0:
                vitals["temperature"] = temp

        # 5. O2 Saturation: e.g. "O2 88%", "spo2 94%", "satting 91%"
        o2_match = re.search(
            r"\b(?:o2|spo2|o2\s+sat|saturation|satting)\s*:?\s*(\d{2,3})\s*%?\b",
            text,
            re.IGNORECASE,
        )
        if o2_match:
            o2 = float(o2_match.group(1))
            if 50 <= o2 <= 100:
                vitals["o2sat"] = o2

        # 6. Pain Scale: e.g. "pain 9/10", "pain 8 out of 10", "pain level 7"
        pain_match = re.search(
            r"\bpain\s*(?:level|score)?\s*:?\s*(\d{1,2})\s*(?:/10|\s*out\s+of\s+10)?\b",
            text,
            re.IGNORECASE,
        )
        if pain_match:
            pain = float(pain_match.group(1))
            if 0 <= pain <= 10:
                vitals["pain"] = pain

        # 7. Age: e.g. "54 year old", "58-year-old", "54yo", "54 y/o", "age: 62"
        age_match = re.search(
            r"\b(?:age\s*:?\s*)?(\d{1,3})\s*[- ]?\s*(?:years?[- ]?old|yo|y/o|yr|yrs)\b",
            text,
            re.IGNORECASE,
        )
        if age_match:
            age = float(age_match.group(1))
            if 0 <= age <= 120:
                vitals["age"] = age
        elif re.search(r"\bage\s*:?\s*(\d{1,3})\b", text, re.IGNORECASE):
            m = re.search(r"\bage\s*:?\s*(\d{1,3})\b", text, re.IGNORECASE)
            age = float(m.group(1))
            if 0 <= age <= 120:
                vitals["age"] = age

        # 8. Gender: e.g. "male", "female", "man", "woman", "M", "F"
        if re.search(r"\b(?:female|woman|lady|girl|she|her)\b", text, re.IGNORECASE):
            vitals["gender"] = "F"
        elif re.search(r"\b(?:male|man|gentleman|boy|he|him)\b", text, re.IGNORECASE):
            vitals["gender"] = "M"
        elif re.search(r"\b\d{1,3}\s*(?:yo|y/o)\s+([mf])\b", text, re.IGNORECASE):
            m = re.search(r"\b\d{1,3}\s*(?:yo|y/o)\s+([mf])\b", text, re.IGNORECASE)
            vitals["gender"] = m.group(1).upper()

        return vitals

    def detect_urgency_level(self, text: str) -> Tuple[str, List[str]]:
        """Assess text urgency level based on critical clinical keywords."""
        detected_keywords = []
        for level in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            for regex in self.urgency_regexes[level]:
                matches = regex.findall(text)
                if matches:
                    detected_keywords.extend(matches)
                    return level, detected_keywords

        return "MODERATE", detected_keywords

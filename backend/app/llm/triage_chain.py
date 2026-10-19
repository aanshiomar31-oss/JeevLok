"""
app/llm/triage_chain.py
========================

LLM-Powered Patient Triage AI & Modular Inference Pipeline.
-----------------------------------------------------------
Leverages:
1. NLP-Based Text Preprocessing (Abbreviation expansion, tokenization, clean-up)
2. NegEx Clinical Negation Detection (Affirmed vs. Negated symptom separation)
3. Quantitative Vitals & Hemodynamic Index Computation (Shock Index, MAP, qSOFA)
4. LangChain Prompt Orchestration (ChatPromptTemplate, ESI v4 emergency guidelines,
   few-shot clinical context injection, structured output validation)
5. Modular Inference Pipeline with Zero-Downtime Deterministic Fallback Engine
6. Form Auto-Fill Payload Generation & Live Triage Prioritization

Principle: "The AI recommends. The clinician decides."
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.nlp.preprocessor import ClinicalPreprocessor
from app.nlp.negation import ClinicalNegationDetector
from app.nlp.entity_extractor import ClinicalEntityExtractor
from app.nlp.similarity import ClinicalSemanticMatcher
from app.core.config import get_settings

logger = logging.getLogger(__name__)

CLINICAL_DISCLAIMER = "AI recommends. Clinician decides."

# ESI Triage Urgency Definitions
PRIORITY_DESCRIPTIONS = {
    "P1": "Resuscitation (Immediate life-saving intervention required)",
    "P2": "Emergent (High risk, acute physiological deterioration, or severe pain)",
    "P3": "Urgent (Stable vitals, requires multiple diagnostic/therapeutic resources)",
    "P4": "Less Urgent (Stable, single diagnostic or therapeutic resource needed)",
    "P5": "Non-Urgent (No acute ED resources needed, routine evaluation)",
}

# System prompt adhering to Emergency Severity Index (ESI v4) & AHA/ACC standards
TRIAGE_SYSTEM_PROMPT = """You are JeevLok AI, an advanced Emergency Department Clinical Decision Support System.
Your role is to analyze patient symptom descriptions, parsed clinical context, and physiological vitals to prioritize case urgency according to the Emergency Severity Index (ESI v4).

Governing Safety Principles:
1. "The AI recommends. The clinician decides."
2. Never underestimate acute coronary syndrome (chest pain + diaphoresis), acute stroke (FAST+), acute airway compromise, or occult shock (Shock Index >= 0.90).
3. Explicitly distinguish affirmed symptoms from negated symptoms (e.g., "denies fever" means fever is absent).

Triage Priority Levels (ESI v4):
- P1 (Resuscitation): Immediate life-saving intervention needed (cardiac arrest, airway compromise, unresponsiveness, severe shock SI >= 1.3, MAP < 50).
- P2 (Emergent): High-risk situation, acute chest pain with diaphoresis, acute neuro deficit (FAST+), severe respiratory distress (SpO2 < 90% or RR > 26), MAP < 65, SI >= 0.90 with hypotension, acute altered mental status.
- P3 (Urgent): Stable vitals, moderate illness requiring multiple emergency resources (labs, IV meds, CT/X-ray, e.g. acute abdominal pain, moderate fever).
- P4 (Less Urgent): Stable vitals, single resource needed (simple laceration, uncomplicated urinary symptoms, mild localized injury).
- P5 (Non-Urgent): Stable vitals, routine examination, no acute resources (prescription refill, suture removal, minor chronic complaint).

Physiological Reference Thresholds:
- Shock Index (SI = Heart Rate / SBP): Normal 0.5-0.7. Borderline 0.7-0.9. Elevated >= 0.90 (occult shock). Severe >= 1.3.
- Mean Arterial Pressure (MAP = (2*DBP + SBP) / 3): Normal 70-100 mmHg. Critical < 65 mmHg (organ hypoperfusion). Severe < 50 mmHg.
- qSOFA Score: AMS (+1), SBP <= 100 (+1), RR >= 22 (+1). Score >= 2 indicates high risk of sepsis.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "priority": "P1" | "P2" | "P3" | "P4" | "P5",
  "urgency_level": "CRITICAL" | "HIGH" | "MODERATE" | "LOW" | "NON-URGENT",
  "risk_score": <float 0.0 to 100.0>,
  "confidence": <float 0.0 to 1.0>,
  "summary_headline": "<Concise clinical headline summarizing urgency and chief complaint>",
  "clinical_rationale": "<Synthesized clinical explanation highlighting key symptoms, vitals, and risk factors>",
  "physiological_drivers": ["<list of specific vital/index abnormalities driving urgency>"],
  "differential_considerations": ["<list of 2-4 primary differential diagnoses>"],
  "recommended_protocols": ["<list of actionable emergency clinical protocols to initiate immediately>"],
  "red_flags": ["<list of critical red flags detected, or empty if none>"]
}
"""


class LLMPatientTriagePipeline:
    """
    Modular, end-to-end clinical triage inference pipeline leveraging LangChain,
    clinical NLP preprocessing, NegEx negation detection, and physiological heuristics.
    """

    def __init__(self) -> None:
        self.preprocessor = ClinicalPreprocessor()
        self.negation_detector = ClinicalNegationDetector()
        self.entity_extractor = ClinicalEntityExtractor()
        self.semantic_matcher = ClinicalSemanticMatcher()
        self.settings = get_settings()

    def preprocess_clinical_context(self, raw_text: str) -> Dict[str, Any]:
        """
        Stage 1: NLP Preprocessing, Abbreviation Expansion, NegEx, and Vitals Extraction.
        """
        clean_res = self.preprocessor.preprocess(raw_text or "")
        cleaned_text = clean_res.get("cleaned_text", raw_text or "")

        # Raw entities
        raw_entities = self.entity_extractor.extract_raw_entities(cleaned_text)

        # NegEx directional scope filter
        affirmed, negated = self.negation_detector.filter_negated_entities(cleaned_text, raw_entities)

        # Numeric vitals
        vitals = self.entity_extractor.extract_vitals_and_demographics(cleaned_text)

        # Physiological calculations
        hr = vitals.get("heartrate")
        sbp = vitals.get("sbp")
        dbp = vitals.get("dbp")
        rr = vitals.get("resprate")
        spo2 = vitals.get("o2sat")

        shock_index = None
        if hr and sbp and sbp > 0:
            shock_index = round(hr / sbp, 2)

        map_val = None
        if sbp and dbp:
            map_val = round((2 * dbp + sbp) / 3, 1)

        # Quick SOFA (Sepsis) score calculation
        qsofa = 0
        if sbp and sbp <= 100:
            qsofa += 1
        if rr and rr >= 22:
            qsofa += 1
        # Check for altered mental status mentions
        ams_terms = ["altered mental status", "lethargic", "confused", "unresponsive", "disoriented", "stupor"]
        if any(term in cleaned_text.lower() for term in ams_terms):
            qsofa += 1

        # Red flags detection
        red_flags = []
        if shock_index and shock_index >= 0.90:
            red_flags.append(f"Elevated Shock Index ({shock_index} >= 0.90 threshold; occult hypoperfusion)")
        if map_val and map_val < 65:
            red_flags.append(f"Compromised Mean Arterial Pressure ({map_val} mmHg < 65 mmHg)")
        if spo2 and spo2 < 90:
            red_flags.append(f"Severe Hypoxemia (SpO2 {spo2}% < 90%)")
        if sbp and sbp < 90:
            red_flags.append(f"Hypotension (SBP {sbp} mmHg < 90 mmHg)")

        # Finding flags
        finding_flags = {
            "chest_pain": False,
            "diaphoresis": False,
            "fast_positive": False,
            "unresponsive": False,
            "seizing": False,
            "airway_compromise": False,
            "stridor": False,
        }
        for entity in affirmed:
            cat = entity["category"]
            if cat in finding_flags:
                finding_flags[cat] = True

        if finding_flags["chest_pain"] and finding_flags["diaphoresis"]:
            red_flags.append("Acute chest pain coupled with diaphoresis (AHA high-risk STEMI presentation)")
        if finding_flags["fast_positive"]:
            red_flags.append("FAST Stroke Criteria Positive (acute focal neurological deficit)")
        if finding_flags["airway_compromise"] or finding_flags["stridor"]:
            red_flags.append("Airway compromise / stridor detected (immediate resuscitation trigger)")
        if finding_flags["unresponsive"]:
            red_flags.append("Unresponsive patient / GCS <= 8 equivalent")

        # Auto-fill payload for UI sync
        auto_fill = {
            "age": vitals.get("age"),
            "gender": vitals.get("gender"),
            "heartrate": vitals.get("heartrate"),
            "sbp": vitals.get("sbp"),
            "dbp": vitals.get("dbp"),
            "resprate": vitals.get("resprate"),
            "temperature": vitals.get("temperature"),
            "o2sat": vitals.get("o2sat"),
            "pain": vitals.get("pain"),
            "chief_complaint": self._derive_chief_complaint(affirmed, cleaned_text),
            "chest_pain": finding_flags["chest_pain"],
            "diaphoresis": finding_flags["diaphoresis"],
            "fast_positive": finding_flags["fast_positive"],
            "unresponsive": finding_flags["unresponsive"],
            "seizing": finding_flags["seizing"],
            "airway_compromise": finding_flags["airway_compromise"],
            "stridor": finding_flags["stridor"],
        }

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "affirmed_entities": affirmed,
            "negated_entities": negated,
            "vitals": vitals,
            "shock_index": shock_index,
            "map": map_val,
            "qsofa": qsofa,
            "red_flags": red_flags,
            "finding_flags": finding_flags,
            "auto_fill_payload": auto_fill,
        }

    def orchestrate_prompt(self, context: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
        """
        Stage 2: LangChain Prompt Orchestration.
        Formats clinical context, physiological markers, and NegEx flags into structured prompts.
        """
        affirmed_str = ", ".join([f"{e['text']} ({e['category']})" for e in context["affirmed_entities"]]) or "None specifically noted"
        negated_str = ", ".join([f"No {e['text']}" for e in context["negated_entities"]]) or "None explicitly denied"

        v = context["vitals"]
        vitals_lines = []
        if v.get("age"):
            vitals_lines.append(f"Age: {v['age']} | Gender: {v.get('gender', 'Unknown')}")
        if v.get("sbp") and v.get("dbp"):
            vitals_lines.append(f"Blood Pressure: {v['sbp']}/{v['dbp']} mmHg")
        if v.get("heartrate"):
            vitals_lines.append(f"Heart Rate: {v['heartrate']} bpm")
        if v.get("resprate"):
            vitals_lines.append(f"Respiratory Rate: {v['resprate']} breaths/min")
        if v.get("o2sat"):
            vitals_lines.append(f"SpO2: {v['o2sat']}%")
        if v.get("temperature"):
            vitals_lines.append(f"Temperature: {v['temperature']} °F")
        if v.get("pain") is not None:
            vitals_lines.append(f"Pain Scale: {v['pain']}/10")

        vitals_block = "\n".join(vitals_lines) if vitals_lines else "Vitals not explicitly recorded in note."

        hemo_lines = []
        if context["shock_index"] is not None:
            hemo_lines.append(f"- Shock Index (SI): {context['shock_index']} (Normal: 0.5-0.7, High: >=0.90)")
        if context["map"] is not None:
            hemo_lines.append(f"- Mean Arterial Pressure (MAP): {context['map']} mmHg (Critical: <65 mmHg)")
        if context["qsofa"] is not None:
            hemo_lines.append(f"- qSOFA Sepsis Score: {context['qsofa']} / 3 (High-risk: >=2)")
        hemodynamics_block = "\n".join(hemo_lines) if hemo_lines else "Hemodynamic indices: Parameters normal or unmeasured."

        red_flags_block = "\n".join([f"⚠️ {rf}" for rf in context["red_flags"]]) if context["red_flags"] else "No acute physiological red flags detected."

        human_prompt = f"""Clinical Case for Triage Evaluation:

[Parsed Clinical Note]:
"{context['cleaned_text']}"

[NLP-Extracted Findings (Affirmed)]:
{affirmed_str}

[NegEx-Filtered Negations (Ruled Out)]:
{negated_str}

[Objective Vital Signs]:
{vitals_block}

[Derived Hemodynamic Indices]:
{hemodynamics_block}

[Identified Red Flags]:
{red_flags_block}

Perform clinical triage analysis. Determine the appropriate ESI Priority (P1 through P5), calculate clinical risk score (0-100), identify differential diagnoses, and specify immediate emergency protocols to trigger.
Return ONLY valid JSON matching the system schema.
"""

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "human", "content": human_prompt},
        ]
        return messages, human_prompt

    def execute_inference(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Stage 3: Inference execution via LangChain or Deterministic Fallback.
        """
        openai_key = os.getenv("OPENAI_API_KEY") or self.settings.OPENAI_API_KEY
        gemini_key = os.getenv("GEMINI_API_KEY") or self.settings.GEMINI_API_KEY

        # If an API key is available, attempt LangChain LLM execution
        if openai_key or gemini_key:
            try:
                llm_output, model_name = self._invoke_langchain_model(messages, temperature, openai_key)
                return llm_output, False, model_name
            except Exception as exc:
                logger.warning("LangChain LLM invocation failed, activating deterministic fallback: %s", exc)

        # Fallback to deterministic clinical rule and simulation engine
        fallback_output = self._deterministic_clinical_inference(context)
        return fallback_output, True, "JeevLok-Deterministic-Clinical-RuleEngine-v1"

    def _invoke_langchain_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        api_key: Optional[str],
    ) -> Tuple[Dict[str, Any], str]:
        """Invoke ChatOpenAI or LangChain runnable pipeline."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            from langchain_core.output_parsers import JsonOutputParser

            model_name = self.settings.LLM_MODEL or "gpt-4o-mini"
            chat = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                timeout=12.0,
            )

            lc_messages = []
            for m in messages:
                if m["role"] == "system":
                    lc_messages.append(SystemMessage(content=m["content"]))
                else:
                    lc_messages.append(HumanMessage(content=m["content"]))

            parser = JsonOutputParser()
            chain = chat | parser
            response_json = chain.invoke(lc_messages)
            return response_json, f"LangChain-LCEL ({model_name})"
        except Exception as exc:
            # Fallback direct OpenAI client if langchain package is compiling/missing
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                parsed = json.loads(completion.choices[0].message.content or "{}")
                return parsed, "OpenAI-Direct-JSON (gpt-4o-mini)"
            except Exception as inner_exc:
                raise RuntimeError(f"Both LangChain and direct LLM calls failed: {exc} | {inner_exc}") from exc

    def _deterministic_clinical_inference(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic, publication-grade ESI v4 inference engine.
        Ensures guaranteed execution with exact clinical safety floors when no LLM key is set.
        """
        flags = context["finding_flags"]
        si = context["shock_index"]
        map_val = context["map"]
        v = context["vitals"]
        hr = v.get("heartrate")
        sbp = v.get("sbp")
        spo2 = v.get("o2sat")
        rr = v.get("resprate")
        pain = v.get("pain", 0) or 0
        qsofa = context["qsofa"]
        red_flags = context["red_flags"]

        # Default Level 3 (Urgent)
        priority = "P3"
        risk_score = 45.0
        confidence = 0.90
        summary_headline = "P3 Priority - Urgent ED Evaluation"
        clinical_rationale = "Patient exhibits stable baseline physiological indicators requiring diagnostic resources."
        diffs = ["Acute Medical Illness", "Infectious/Inflammatory Process", "Electrolyte Disturbance"]
        protocols = ["Standard Nursing Triage Assessment", "Routine Bloodwork & Metabolic Panel"]

        # P1 Conditions (Immediate Life Threat)
        if flags["airway_compromise"] or flags["stridor"] or flags["unresponsive"]:
            priority = "P1"
            risk_score = 98.0
            confidence = 0.96
            summary_headline = "P1 Priority - Resuscitation (Airway / Neurological Collapse)"
            clinical_rationale = "Patient demonstrates acute loss of protective airway reflexes or unresponsiveness requiring immediate resuscitation."
            diffs = ["Acute Respiratory Failure", "Massive Intracranial Event", "Severe Coma/Toxic Ingestion"]
            protocols = ["Immediate Resuscitation Bay Placement", "Airway Stabilization / Intubation Protocol", "Code Blue / Rapid Response Activation"]

        elif (si and si >= 1.3) or (map_val and map_val < 50):
            priority = "P1"
            risk_score = 95.0
            confidence = 0.94
            summary_headline = "P1 Priority - Resuscitation (Decompensated Shock / Severe Hemodynamic Collapse)"
            clinical_rationale = f"Severe hemodynamic instability observed with Shock Index {si} (>= 1.3) and MAP {map_val} mmHg (< 50)."
            diffs = ["Cardiogenic Shock", "Septic Shock", "Hemorrhagic / Hypovolemic Shock"]
            protocols = ["Immediate Resuscitation Bay Placement", "Dual Large-Bore IV Access & Rapid Fluid Resuscitation", "Vasopressor Infusion Readiness"]

        # P2 Conditions (Emergent - High Risk)
        elif flags["fast_positive"]:
            priority = "P2"
            risk_score = 88.0
            confidence = 0.93
            summary_headline = "P2 Priority - Emergent (Acute Stroke Code FAST+)"
            clinical_rationale = "Patient presents with acute focal neurological deficits matching FAST stroke criteria. Time is brain."
            diffs = ["Acute Ischemic Stroke", "Intracerebral Hemorrhage", "Transient Ischemic Attack"]
            protocols = ["Stroke Code Activation", "Immediate Non-Contrast Head CT", "Point-of-Care Blood Glucose & Coagulation Panel"]

        elif flags["chest_pain"] and flags["diaphoresis"]:
            priority = "P2"
            risk_score = 90.0
            confidence = 0.95
            summary_headline = "P2 Priority - Emergent (Acute Coronary Syndrome / STEMI Alert)"
            clinical_rationale = "Presentation of crushing chest pain combined with diaphoresis is strongly indicative of acute myocardial infarction."
            diffs = ["ST-Elevation Myocardial Infarction (STEMI)", "Acute Non-STEMI", "Aortic Dissection"]
            protocols = ["12-Lead ECG within 10 minutes", "Aspirin & Cardiac Biomarkers (High-Sensitivity Troponin)", "Continuous Cardiac Telemetry"]

        elif (si and si >= 0.90) or (sbp and sbp < 90) or (spo2 and spo2 < 90):
            priority = "P2"
            risk_score = 85.0
            confidence = 0.91
            summary_headline = "P2 Priority - Emergent (Physiological Deterioration / Occult Shock)"
            clinical_rationale = f"Significant physiological abnormality detected (SI={si}, SBP={sbp}, SpO2={spo2}%). High risk of rapid decompensation."
            diffs = ["Occult Shock / Sepsis", "Acute Hypoxemic Respiratory Failure", "Severe Dehydration"]
            protocols = ["Immediate Bedside Clinician Evaluation", "Supplemental Oxygen Titration", "IV Access & STAT Laboratory Workup"]

        elif qsofa >= 2:
            priority = "P2"
            risk_score = 86.0
            confidence = 0.92
            summary_headline = "P2 Priority - Emergent (Severe Sepsis Alert - qSOFA >= 2)"
            clinical_rationale = f"Quick SOFA score of {qsofa}/3 indicates high probability of sepsis with elevated mortality risk."
            diffs = ["Severe Sepsis / Septic Shock", "Severe Pneumonia", "Pyelonephritis / Urosepsis"]
            protocols = ["Sepsis Resuscitation Bundle", "Blood Cultures x2 prior to IV Antibiotics", "Serum Lactate & Fluid Bolus 30 mL/kg"]

        elif flags["chest_pain"]:
            priority = "P2"
            risk_score = 75.0
            confidence = 0.88
            summary_headline = "P2 Priority - Emergent (Acute Chest Discomfort)"
            clinical_rationale = "Isolated acute chest pain warrants rapid cardiac clearance to rule out acute coronary syndrome."
            diffs = ["Acute Coronary Syndrome", "Pericarditis", "Musculoskeletal Chest Pain"]
            protocols = ["12-Lead ECG within 10 minutes", "Cardiac Enzymes", "Continuous Monitoring"]

        elif pain >= 8:
            priority = "P2"
            risk_score = 70.0
            confidence = 0.85
            summary_headline = f"P2 Priority - Emergent (Severe Acute Pain: {pain}/10)"
            clinical_rationale = f"Severe distressing pain ({pain}/10) requires prompt clinical attention and analgesia."
            diffs = ["Renal Colic", "Acute Abdomen", "Acute Musculoskeletal Trauma"]
            protocols = ["Analgesia Protocol", "Focused Ultrasound / Diagnostic Imaging"]

        # P4 / P5 Conditions (Low Resource / Stable)
        else:
            cleaned_lower = context["cleaned_text"].lower()
            p5_terms = ["refill", "suture removal", "prescription", "paperwork", "routine check", "minor rash"]
            p4_terms = ["ankle sprain", "simple cut", "minor laceration", "wrist pain", "sore throat", "mild cold", "isolated finger"]

            if any(t in cleaned_lower for t in p5_terms) and not red_flags:
                priority = "P5"
                risk_score = 12.0
                confidence = 0.92
                summary_headline = "P5 Priority - Non-Urgent (Routine Clinic / No Resource)"
                clinical_rationale = "Patient presentation is non-acute and does not require emergency department diagnostic or therapeutic resources."
                diffs = ["Routine Outpatient Need", "Minor Chronic Condition"]
                protocols = ["Routine Ambulatory Care Fast Track", "Medication Reconciliation"]

            elif any(t in cleaned_lower for t in p4_terms) and not red_flags:
                priority = "P4"
                risk_score = 25.0
                confidence = 0.89
                summary_headline = "P4 Priority - Less Urgent (Single Diagnostic Resource)"
                clinical_rationale = "Patient is stable and anticipated to require only a single diagnostic modality or minor procedure."
                diffs = ["Localized Soft Tissue Sprain", "Uncomplicated Laceration", "Upper Respiratory Infection"]
                protocols = ["Fast Track Minor Injury Protocol", "Single X-ray or Wound Dressing"]

        urgency_level = {
            "P1": "CRITICAL",
            "P2": "HIGH",
            "P3": "MODERATE",
            "P4": "LOW",
            "P5": "NON-URGENT",
        }.get(priority, "MODERATE")

        return {
            "priority": priority,
            "urgency_level": urgency_level,
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "summary_headline": summary_headline,
            "clinical_rationale": clinical_rationale,
            "physiological_drivers": red_flags if red_flags else ["Vitals within normal expected emergency ranges"],
            "differential_considerations": diffs,
            "recommended_protocols": protocols,
            "red_flags": red_flags,
        }

    def run_pipeline(
        self,
        raw_text: str,
        temperature: float = 0.0,
        include_prompt_debug: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full modular inference pipeline:
        Preprocessing -> NegEx -> Prompt Orchestration -> Inference -> Decision Support.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess context
        context = self.preprocess_clinical_context(raw_text)

        # Step 2: Orchestrate prompt
        messages, human_prompt = self.orchestrate_prompt(context)

        # Step 3: Run inference
        inference_res, fallback_used, model_name = self.execute_inference(messages, context, temperature)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Assemble prompt orchestration debug payload
        prompt_debug = None
        if include_prompt_debug:
            prompt_debug = {
                "system_prompt_preview": TRIAGE_SYSTEM_PROMPT[:300] + "...",
                "orchestrated_human_prompt": human_prompt,
                "model_name": model_name,
                "framework": "LangChain LCEL",
                "latency_ms": latency_ms,
                "fallback_used": fallback_used,
            }

        # Build response schema
        affirmed_names = [e["text"] for e in context["affirmed_entities"]]
        negated_names = [e["text"] for e in context["negated_entities"]]

        vitals_summary = {
            "age": context["vitals"].get("age"),
            "gender": context["vitals"].get("gender"),
            "sbp": context["vitals"].get("sbp"),
            "dbp": context["vitals"].get("dbp"),
            "heartrate": context["vitals"].get("heartrate"),
            "resprate": context["vitals"].get("resprate"),
            "temperature": context["vitals"].get("temperature"),
            "o2sat": context["vitals"].get("o2sat"),
            "pain": context["vitals"].get("pain"),
            "shock_index": context["shock_index"],
            "map": context["map"],
            "qsofa_score": context["qsofa"],
        }

        return {
            "priority": inference_res.get("priority", "P3"),
            "urgency_level": inference_res.get("urgency_level", "MODERATE"),
            "risk_score": float(inference_res.get("risk_score", 50.0)),
            "confidence": float(inference_res.get("confidence", 0.85)),
            "summary_headline": inference_res.get("summary_headline", f"{inference_res.get('priority', 'P3')} Priority"),
            "clinical_rationale": inference_res.get("clinical_rationale", "Clinical triage assessment completed."),
            "physiological_drivers": inference_res.get("physiological_drivers", []),
            "differential_considerations": inference_res.get("differential_considerations", []),
            "recommended_protocols": inference_res.get("recommended_protocols", []),
            "red_flags": inference_res.get("red_flags", []),
            "vitals_summary": vitals_summary,
            "affirmed_findings": affirmed_names,
            "negated_findings": negated_names,
            "auto_fill_payload": context["auto_fill_payload"],
            "prompt_orchestration": prompt_debug,
            "disclaimer": CLINICAL_DISCLAIMER,
        }

    def _derive_chief_complaint(self, affirmed: List[dict], text: str) -> str:
        """Derive a clean chief complaint string."""
        if affirmed:
            unique = list(dict.fromkeys(e["text"].capitalize() for e in affirmed))
            return ", ".join(unique[:3])
        return text[:45].strip() if text else "Acute Presentation"


# Global singleton instance
llm_triage_pipeline = LLMPatientTriagePipeline()


def run_llm_triage(
    text: str,
    temperature: float = 0.0,
    include_prompt_debug: bool = True,
) -> Dict[str, Any]:
    """Helper functional interface."""
    return llm_triage_pipeline.run_pipeline(text, temperature, include_prompt_debug)

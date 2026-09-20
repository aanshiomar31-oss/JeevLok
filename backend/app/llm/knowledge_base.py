"""
app/llm/knowledge_base.py
========================

Curated Emergency Department Clinical Knowledge Base for RAG Retrieval.
Contains validated medical protocols, physiological formulas, reference ranges,
and Emergency Severity Index (ESI) guidelines.
"""

from __future__ import annotations

from typing import Dict, List, Any

CLINICAL_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "esi_triage_levels",
        "title": "Emergency Severity Index (ESI) Triage Levels",
        "keywords": ["esi", "priority", "p1", "p2", "p3", "p4", "p5", "triage levels", "acuity"],
        "content": (
            "The Emergency Severity Index (ESI) is a five-level emergency triage algorithm: "
            "P1 (Resuscitation): Immediate life-saving intervention needed (unresponsive, severe respiratory distress, cardiac arrest, intubated). "
            "P2 (Emergent): High-risk situation, altered mental status, severe pain/distress, or vitals in dangerous zone (e.g. SpO2 < 92%, HR > 120, SBP < 90). "
            "P3 (Urgent): Clinically stable, but requires two or more hospital resources (e.g., blood labs, IV fluids, CT scan, X-ray). "
            "P4 (Less Urgent): Clinically stable, requires exactly one diagnostic or therapeutic resource (e.g., simple X-ray, oral medication). "
            "P5 (Non-Urgent): Clinically stable, requires no specialized emergency resources (e.g., prescription refill, simple suture removal)."
        ),
    },
    {
        "id": "shock_index",
        "title": "Shock Index (SI) Clinical Significance",
        "keywords": ["shock index", "si", "hypovolemia", "hypoperfusion", "heart rate", "sbp", "vital signs"],
        "content": (
            "Shock Index (SI) is calculated as Heart Rate divided by Systolic Blood Pressure (SI = HR / SBP). "
            "Normal range: 0.5 to 0.7. "
            "An elevated Shock Index (≥ 0.9) is a sensitive early marker for occult shock, internal hemorrhage, "
            "sepsis, or acute ventricular decompensation—frequently detecting physiological instability "
            "before blood pressure visibly collapses (compensatory tachycardia maintains BP initially). "
            "In trauma or acute medical patients, SI ≥ 1.0 indicates severe hemodynamic instability and warrants immediate resuscitation."
        ),
    },
    {
        "id": "map_formula",
        "title": "Mean Arterial Pressure (MAP) Reference and Calculation",
        "keywords": ["map", "mean arterial pressure", "blood pressure", "perfusion", "sbp", "dbp"],
        "content": (
            "Mean Arterial Pressure (MAP) represents the average perfusion pressure across the cardiac cycle. "
            "Formula: MAP = (2 * DBP + SBP) / 3. "
            "Normal resting range: 70 to 100 mmHg. "
            "A MAP < 65 mmHg is a critical clinical threshold indicating inadequate organ perfusion, placing the brain, "
            "kidneys, and myocardium at risk of ischemic injury. Patients with MAP < 65 mmHg should be triaged as P1 or P2 "
            "for emergent fluid resuscitation or vasopressor evaluation."
        ),
    },
    {
        "id": "pulse_pressure",
        "title": "Pulse Pressure (PP) Clinical Interpretation",
        "keywords": ["pulse pressure", "pp", "sbp", "dbp", "narrow pulse pressure"],
        "content": (
            "Pulse Pressure is the difference between Systolic and Diastolic Blood Pressure (PP = SBP - DBP). "
            "Normal value: ~40 mmHg. "
            "A narrow pulse pressure (< 25 mmHg) indicates decreased stroke volume, frequently observed in hypovolemic shock, "
            "cardiogenic shock, cardiac tamponade, or severe aortic stenosis. "
            "A wide pulse pressure (> 60 mmHg) is seen in septic shock (hyperdynamic state), aortic regurgitation, or severe arterial stiffness."
        ),
    },
    {
        "id": "sepsis_screening",
        "title": "Sepsis Screening (qSOFA & SIRS Guidelines)",
        "keywords": ["sepsis", "qsofa", "sirs", "infection", "fever", "respiratory rate", "hypotension"],
        "content": (
            "Quick Sequential Organ Failure Assessment (qSOFA) identifies emergency patients with suspected infection "
            "at high risk of in-hospital deterioration. Criteria (1 point each, score ≥ 2 is high risk): "
            "1. Respiratory rate ≥ 22 breaths/min. "
            "2. Altered mentation (GCS < 15 or acute confusion). "
            "3. Systolic blood pressure ≤ 100 mmHg. "
            "SIRS criteria include temperature (> 38.3°C or < 36°C), heart rate > 90 bpm, and respiratory rate > 20/min. "
            "High-risk sepsis triggers immediate IV fluid bolus, blood cultures, and broad-spectrum antibiotics within 1 hour."
        ),
    },
    {
        "id": "stemi_protocol",
        "title": "STEMI (ST-Elevation Myocardial Infarction) Protocol",
        "keywords": ["stemi", "chest pain", "angina", "heart attack", "ecg", "troponin", "diaphoresis"],
        "content": (
            "Patients presenting with acute chest pain, chest tightness, or substernal discomfort accompanied by diaphoresis, "
            "radiation to the left arm/jaw, or dyspnea require an immediate 12-lead ECG performed and interpreted within "
            "10 minutes of arrival. If ST-elevation is confirmed, activate the cardiac catheterization lab immediately for primary PCI."
        ),
    },
    {
        "id": "stroke_protocol",
        "title": "Acute Ischemic Stroke (FAST Protocol)",
        "keywords": ["stroke", "fast", "facial droop", "slurred speech", "arm weakness", "ct head", "tpa"],
        "content": (
            "The FAST assessment evaluates: "
            "F (Face drooping): Ask patient to smile. "
            "A (Arm weakness): Ask patient to raise both arms. "
            "S (Speech difficulty): Ask patient to repeat a simple sentence. "
            "T (Time to call): Record exact 'last known well' time. "
            "Patients presenting within the 4.5-hour thrombolytic window require emergent non-contrast head CT to rule out hemorrhage."
        ),
    },
    {
        "id": "mimic_iv_ed_dataset",
        "title": "MIMIC-IV-ED Dataset and Machine Learning Baseline",
        "keywords": ["mimic", "dataset", "physionet", "mit", "training", "ensemble"],
        "content": (
            "The ClinicalPulse AI models are trained on the MIMIC-IV-ED database from Beth Israel Deaconess Medical Center, "
            "published through MIT PhysioNet. The model architecture combines a deterministic Clinical Rule Engine "
            "with a calibrated Gradient Boosting Ensemble (XGBoost, LightGBM, CatBoost, HistGradientBoosting) to produce "
            "both risk scores and SHAP explainability values for emergency clinicians."
        ),
    },
]

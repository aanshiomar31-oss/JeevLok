"""
schemas/voice.py
================

Pydantic schemas for the Voice-to-Triage speech-to-text endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class VoiceTranscribeResponse(BaseModel):
    """Transcription response with integrated Clinical NLP auto-fill payload."""
    transcript: str = Field(..., description="Transcribed clinical text from the audio input.")
    duration_seconds: Optional[float] = None
    language: str = "en"
    parsed_triage: Optional[Dict[str, Any]] = Field(
        None,
        description="Pre-parsed triage features (vitals, findings, auto_fill_payload) extracted from the transcript."
    )

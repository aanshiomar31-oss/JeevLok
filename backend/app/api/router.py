"""
api/router.py
===============

PatientTriage.ai — API Router Aggregation
---------------------------------------------
Single place that wires every route module into one `APIRouter`, which
`app/main.py` mounts under `settings.API_V1_PREFIX`. Adding a new route
module in future milestones means: write the module, import it here,
`include_router` it — nothing in `main.py` changes.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    audit,
    chat,
    explain,
    health,
    model,
    nlp,
    override,
    queue,
    summary,
    triage,
    triage_stays,
    vitals,
    voice,
)
from app.api import security

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(triage_stays.router)
api_router.include_router(model.router)
api_router.include_router(triage.router)
api_router.include_router(queue.router)
api_router.include_router(override.router)
api_router.include_router(vitals.router)
api_router.include_router(audit.router)
api_router.include_router(security.router)

# Enhanced AI + NLP + GenAI routers
api_router.include_router(nlp.router)
api_router.include_router(chat.router)
api_router.include_router(summary.router)
api_router.include_router(voice.router)
api_router.include_router(explain.router)


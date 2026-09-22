"""
Alert generation service.

Converts risk results that cross configurable thresholds into alert
objects. Delivery is dashboard-only for now (persisted + pushed over
the existing WebSocket connection); the structure leaves room for
email/SMS/push channels to be added later without changing callers.
"""
from __future__ import annotations

import uuid
from typing import Optional

from app.models.schemas import AlertResponse, RiskResult

ALERT_SEVERITY_BY_LEVEL = {
    "LOW": None,          # no alert generated
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "CRITICAL",
}

ALERT_MESSAGES = {
    "MEDIUM": "Elevated impersonation risk detected; increased monitoring recommended.",
    "HIGH": "Potential synthetic voice or impersonation indicators detected.",
    "CRITICAL": "High-confidence impersonation risk detected during active call.",
}


def maybe_generate_alert(call_id: str, risk: RiskResult) -> Optional[AlertResponse]:
    severity = ALERT_SEVERITY_BY_LEVEL.get(risk.risk_level)
    if severity is None:
        return None

    from datetime import datetime, timezone

    return AlertResponse(
        alert_id=f"ALERT-{uuid.uuid4().hex[:8].upper()}",
        call_id=call_id,
        severity=severity,
        message=ALERT_MESSAGES[severity],
        recommended_action=risk.recommended_action,
        status="open",
        created_at=datetime.now(timezone.utc),
    )

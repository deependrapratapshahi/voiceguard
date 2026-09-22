"""
Pydantic schemas defining the API contracts.

All probabilistic outputs are explicitly typed as scores/probabilities,
never as certainties, in line with the project's scientific-integrity
requirements.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------- Detection ----------

class DetectionResult(BaseModel):
    synthetic_probability: float = Field(..., ge=0.0, le=1.0)
    label: str  # LIKELY_REAL | UNCERTAIN | LIKELY_SYNTHETIC
    model_version: str


class SpeakerVerificationResult(BaseModel):
    speaker_similarity: float = Field(..., ge=0.0, le=1.0)
    identity_match: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class ProsodyResult(BaseModel):
    prosody_anomaly: float = Field(..., ge=0.0, le=1.0)
    features: dict


# ---------- Risk ----------

class ContextFactors(BaseModel):
    """Non-audio fraud-risk context for a call. `caller_reputation_score`
    is 1.0 (fully trusted, e.g. a long-standing verified account) down
    to 0.0 (known-bad/blacklisted); callers with no reputation history
    should use the neutral default of 1.0 rather than being penalized
    for lack of data."""

    caller_reputation_score: float = Field(1.0, ge=0.0, le=1.0)
    caller_is_registered_contact: bool = True
    is_new_device: bool = False
    transaction_value: Optional[float] = None
    requested_action: Optional[str] = None
    urgency_indicated: bool = False
    bypass_requested: bool = False  # request to bypass registered callback verification
    privileged_operation: bool = False
    unusual_call_time: bool = False
    historical_fraud_flags: int = 0


class RiskWeights(BaseModel):
    """
    Configurable weight for each of the 10 signals the risk engine
    combines. Defaults sum to 1.0 so a maximally-risky call scores
    100; deployments can retune these (e.g. via environment variables,
    see app/config.py) without any code changes.
    """
    synthetic_weight: float = 0.25
    speaker_weight: float = 0.15
    prosody_weight: float = 0.08
    caller_reputation_weight: float = 0.10
    registered_contact_weight: float = 0.07
    transaction_weight: float = 0.15
    privileged_action_weight: float = 0.06
    urgency_weight: float = 0.05
    bypass_callback_weight: float = 0.06
    historical_fraud_weight: float = 0.03


class RiskResult(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    reasons: List[str]
    recommended_action: str


# ---------- Calls ----------

class CallCreateRequest(BaseModel):
    caller_label: Optional[str] = None
    speaker_id: Optional[str] = None
    is_demo: bool = True


class CallResponse(BaseModel):
    call_id: str
    caller_label: Optional[str]
    speaker_id: Optional[str]
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    is_demo: bool

    class Config:
        from_attributes = True


class CallRiskSnapshot(BaseModel):
    call_id: str
    timestamp: datetime
    synthetic_probability: Optional[float]
    speaker_similarity: Optional[float]
    prosody_anomaly: Optional[float]
    risk_score: float
    risk_level: str
    reasons: List[str] = []
    recommended_action: Optional[str] = None


# ---------- Speakers ----------

class SpeakerCreateRequest(BaseModel):
    speaker_id: str
    display_name: str


class SpeakerResponse(BaseModel):
    speaker_id: str
    display_name: str
    embedding_model_version: str
    created_at: datetime

    class Config:
        from_attributes = True


class SpeakerVerifyRequest(BaseModel):
    call_id: Optional[str] = None


# ---------- Alerts ----------

class AlertResponse(BaseModel):
    alert_id: str
    call_id: str
    severity: str
    message: str
    recommended_action: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class AlertCreateRequest(BaseModel):
    call_id: str
    severity: str
    message: str
    recommended_action: str


# ---------- Models ----------

class ModelInfo(BaseModel):
    component: str
    name: str
    version: str
    is_active: bool
    is_baseline: bool


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    app: str
    env: str
    database: str
    redis: str

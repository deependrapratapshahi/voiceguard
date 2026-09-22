from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import ContextFactors, RiskResult
from app.services.risk_engine import compute_risk, weights_from_settings

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])
settings = get_settings()


@router.post("/evaluate", response_model=RiskResult)
async def evaluate_risk(
    context: ContextFactors,
    synthetic_probability: float | None = None,
    speaker_similarity: float | None = None,
    prosody_anomaly: float | None = None,
):
    """
    Evaluates the contextual fraud-risk engine directly, without
    requiring an audio upload -- useful for testing risk logic, CRM/IVR
    integrations that already have their own detection scores, or
    driving the demo scenarios. Weights come from server configuration
    (see app/config.py / .env); thresholds and factor scaling are
    likewise configurable there.
    """
    weights = weights_from_settings()
    return compute_risk(synthetic_probability, speaker_similarity, prosody_anomaly, context, weights)

"""
Contextual fraud-risk engine.

Combines ten signals into a single, explainable 0-100 risk score:
  1. synthetic speech probability
  2. speaker similarity
  3. prosody anomaly
  4. caller reputation
  5. registered contact status
  6. transaction amount
  7. privileged action
  8. urgency
  9. request to bypass registered callback verification
  10. historical fraud indicator

Each signal is converted to a normalized "risk fraction" in [0, 1],
multiplied by a configurable weight (see RiskWeights /
app/config.py), and summed into a 0-100 score. Every contributing
signal that crosses a notable threshold produces a specific,
human-readable reason -- the score is never a black box.

This is a probabilistic risk indicator, not proof of fraud. Downstream
consumers (API, WebSocket, dashboard) must present it as guidance for
additional verification, never as a fraud determination.
"""
from __future__ import annotations

from app.config import get_settings
from app.models.schemas import ContextFactors, RiskResult, RiskWeights

settings = get_settings()

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

RECOMMENDED_ACTIONS = ("ALLOW", "MONITOR", "SECONDARY_VERIFICATION", "CALLBACK_AND_MFA", "ESCALATE")


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def weights_from_settings() -> RiskWeights:
    return RiskWeights(
        synthetic_weight=settings.SYNTHETIC_WEIGHT,
        speaker_weight=settings.SPEAKER_WEIGHT,
        prosody_weight=settings.PROSODY_WEIGHT,
        caller_reputation_weight=settings.CALLER_REPUTATION_WEIGHT,
        registered_contact_weight=settings.REGISTERED_CONTACT_WEIGHT,
        transaction_weight=settings.TRANSACTION_WEIGHT,
        privileged_action_weight=settings.PRIVILEGED_ACTION_WEIGHT,
        urgency_weight=settings.URGENCY_WEIGHT,
        bypass_callback_weight=settings.BYPASS_CALLBACK_WEIGHT,
        historical_fraud_weight=settings.HISTORICAL_FRAUD_WEIGHT,
    )


# ---------------------------------------------------------------------
# Per-signal risk fractions (each in [0, 1])
# ---------------------------------------------------------------------

def _synthetic_fraction(synthetic_probability: float | None) -> float:
    return _clip(synthetic_probability) if synthetic_probability is not None else 0.0


def _speaker_fraction(speaker_similarity: float | None) -> float:
    # No reference registered -> neutral/uncertain contribution, not a
    # penalty for a signal that was never available.
    if speaker_similarity is None:
        return 0.5
    return _clip(1.0 - speaker_similarity)


def _prosody_fraction(prosody_anomaly: float | None) -> float:
    return _clip(prosody_anomaly) if prosody_anomaly is not None else 0.0


def _caller_reputation_fraction(context: ContextFactors) -> float:
    fraction = 1.0 - _clip(context.caller_reputation_score)
    if context.unusual_call_time:
        fraction = _clip(fraction + 0.2)
    return fraction


def _registered_contact_fraction(context: ContextFactors) -> float:
    if not context.caller_is_registered_contact:
        return 1.0
    return 0.4 if context.is_new_device else 0.0


def _transaction_fraction(context: ContextFactors) -> float:
    if context.transaction_value is None:
        return 0.0
    low, high = settings.TRANSACTION_LOW_THRESHOLD, settings.TRANSACTION_HIGH_THRESHOLD
    if high <= low:
        return 1.0 if context.transaction_value >= high else 0.0
    return _clip((context.transaction_value - low) / (high - low))


def _privileged_action_fraction(context: ContextFactors) -> float:
    return 1.0 if context.privileged_operation else 0.0


def _urgency_fraction(context: ContextFactors) -> float:
    return 1.0 if context.urgency_indicated else 0.0


def _bypass_callback_fraction(context: ContextFactors) -> float:
    return 1.0 if context.bypass_requested else 0.0


def _historical_fraud_fraction(context: ContextFactors) -> float:
    cap = max(settings.HISTORICAL_FRAUD_CAP, 1)
    return _clip(context.historical_fraud_flags / cap)


# ---------------------------------------------------------------------
# Explainable reasons
# ---------------------------------------------------------------------

def _build_reasons(
    synthetic_probability: float | None,
    speaker_similarity: float | None,
    prosody_anomaly: float | None,
    context: ContextFactors,
) -> list[str]:
    reasons: list[str] = []

    if synthetic_probability is not None and synthetic_probability >= 0.65:
        reasons.append("High synthetic speech probability")

    if speaker_similarity is None:
        reasons.append("Speaker verification unavailable (no reference registered)")
    elif speaker_similarity < settings.SPEAKER_UNCERTAIN_LOW:
        reasons.append("Low speaker similarity to reference voice")
    elif speaker_similarity < settings.SPEAKER_UNCERTAIN_HIGH:
        reasons.append("Speaker verification uncertain")

    if prosody_anomaly is not None and prosody_anomaly >= 0.6:
        reasons.append("Elevated prosody/acoustic anomaly")

    if context.caller_reputation_score < 0.5:
        reasons.append("Caller has a poor reputation score")
    if context.unusual_call_time:
        reasons.append("Call occurred at an unusual time")

    if not context.caller_is_registered_contact:
        reasons.append("Caller is not a registered contact")
    elif context.is_new_device:
        reasons.append("Call originated from a new/unrecognized device")

    if context.transaction_value is not None and context.transaction_value >= settings.TRANSACTION_HIGH_THRESHOLD:
        reasons.append("High-value transaction")

    if context.privileged_operation:
        reasons.append("Request involves a privileged operation")

    if context.urgency_indicated:
        reasons.append("Urgency indicators present in the request")

    if context.bypass_requested:
        reasons.append("Caller requested bypass of registered callback")

    if context.historical_fraud_flags > 0:
        reasons.append("Prior fraud indicators on record for this caller/account")

    if not reasons:
        reasons.append("No significant risk indicators detected")

    return reasons


# ---------------------------------------------------------------------
# Level and action mapping
# ---------------------------------------------------------------------

def score_to_level(score: float) -> str:
    if score <= settings.RISK_THRESHOLD_LOW:
        return "LOW"
    if score <= settings.RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    if score <= settings.RISK_THRESHOLD_HIGH:
        return "HIGH"
    return "CRITICAL"


def determine_recommended_action(level: str, context: ContextFactors) -> str:
    """
    Maps a risk level to a recommended action. HIGH-risk calls where the
    caller specifically requested a callback bypass or a privileged
    operation are routed to CALLBACK_AND_MFA (directly counter the
    bypass attempt) rather than generic SECONDARY_VERIFICATION.
    """
    if level == "CRITICAL":
        return "ESCALATE"
    if level == "HIGH":
        if context.bypass_requested or context.privileged_operation:
            return "CALLBACK_AND_MFA"
        return "SECONDARY_VERIFICATION"
    if level == "MEDIUM":
        return "MONITOR"
    return "ALLOW"


def compute_context_risk_summary(context: ContextFactors, weights: RiskWeights) -> float:
    """
    A normalized [0, 1] summary of ONLY the non-audio contextual
    factors (everything except synthetic speech / speaker / prosody),
    weighted-averaged by their configured weights. This does not affect
    the actual risk score (which already weights every factor
    individually in compute_risk) -- it exists purely so dashboards can
    show "how risky is the context alone" as a single number.
    """
    context_weight_total = (
        weights.caller_reputation_weight
        + weights.registered_contact_weight
        + weights.transaction_weight
        + weights.privileged_action_weight
        + weights.urgency_weight
        + weights.bypass_callback_weight
        + weights.historical_fraud_weight
    )
    if context_weight_total <= 0:
        return 0.0

    weighted = (
        _caller_reputation_fraction(context) * weights.caller_reputation_weight
        + _registered_contact_fraction(context) * weights.registered_contact_weight
        + _transaction_fraction(context) * weights.transaction_weight
        + _privileged_action_fraction(context) * weights.privileged_action_weight
        + _urgency_fraction(context) * weights.urgency_weight
        + _bypass_callback_fraction(context) * weights.bypass_callback_weight
        + _historical_fraud_fraction(context) * weights.historical_fraud_weight
    )
    return _clip(weighted / context_weight_total)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def compute_risk(
    synthetic_probability: float | None,
    speaker_similarity: float | None,
    prosody_anomaly: float | None,
    context: ContextFactors,
    weights: RiskWeights,
) -> RiskResult:
    weighted_sum = (
        _synthetic_fraction(synthetic_probability) * weights.synthetic_weight
        + _speaker_fraction(speaker_similarity) * weights.speaker_weight
        + _prosody_fraction(prosody_anomaly) * weights.prosody_weight
        + _caller_reputation_fraction(context) * weights.caller_reputation_weight
        + _registered_contact_fraction(context) * weights.registered_contact_weight
        + _transaction_fraction(context) * weights.transaction_weight
        + _privileged_action_fraction(context) * weights.privileged_action_weight
        + _urgency_fraction(context) * weights.urgency_weight
        + _bypass_callback_fraction(context) * weights.bypass_callback_weight
        + _historical_fraud_fraction(context) * weights.historical_fraud_weight
    )

    score = round(_clip(weighted_sum, 0.0, 1.0) * 100.0, 1)
    level = score_to_level(score)
    reasons = _build_reasons(synthetic_probability, speaker_similarity, prosody_anomaly, context)
    action = determine_recommended_action(level, context)

    return RiskResult(risk_score=score, risk_level=level, reasons=reasons, recommended_action=action)

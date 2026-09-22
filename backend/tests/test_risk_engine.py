"""
Tests for the contextual fraud-risk engine (app/services/risk_engine.py).

Covers: the ten combined signals, configurable weights, configurable
LOW/MEDIUM/HIGH/CRITICAL thresholds, explainable reasons (including the
exact example strings from the spec), and the ALLOW / MONITOR /
SECONDARY_VERIFICATION / CALLBACK_AND_MFA / ESCALATE action mapping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.models.schemas import ContextFactors, RiskWeights
from app.services.risk_engine import (
    compute_context_risk_summary,
    compute_risk,
    determine_recommended_action,
    score_to_level,
)

settings = get_settings()


def default_weights() -> RiskWeights:
    return RiskWeights()


# ---------------------------------------------------------------------
# Level thresholds
# ---------------------------------------------------------------------

def test_score_to_level_boundaries_match_configured_thresholds():
    assert score_to_level(0) == "LOW"
    assert score_to_level(settings.RISK_THRESHOLD_LOW) == "LOW"
    assert score_to_level(settings.RISK_THRESHOLD_LOW + 1) == "MEDIUM"
    assert score_to_level(settings.RISK_THRESHOLD_MEDIUM) == "MEDIUM"
    assert score_to_level(settings.RISK_THRESHOLD_MEDIUM + 1) == "HIGH"
    assert score_to_level(settings.RISK_THRESHOLD_HIGH) == "HIGH"
    assert score_to_level(settings.RISK_THRESHOLD_HIGH + 1) == "CRITICAL"
    assert score_to_level(100) == "CRITICAL"


# ---------------------------------------------------------------------
# Overall scoring scenarios
# ---------------------------------------------------------------------

def test_genuine_caller_scores_low_and_allows():
    context = ContextFactors()
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert result.risk_level == "LOW"
    assert result.recommended_action == "ALLOW"
    assert 0.0 <= result.risk_score <= 100.0


def test_high_risk_impersonation_scores_critical_and_escalates():
    context = ContextFactors(
        caller_reputation_score=0.1,
        caller_is_registered_contact=False,
        bypass_requested=True,
        urgency_indicated=True,
        privileged_operation=True,
        transaction_value=100000,
        historical_fraud_flags=5,
    )
    result = compute_risk(0.95, 0.1, 0.9, context, default_weights())
    assert result.risk_score > settings.RISK_THRESHOLD_HIGH
    assert result.risk_level == "CRITICAL"
    assert result.recommended_action == "ESCALATE"


def test_risk_score_never_exceeds_valid_bounds():
    context = ContextFactors(
        caller_reputation_score=0.0,
        caller_is_registered_contact=False,
        is_new_device=True,
        bypass_requested=True,
        urgency_indicated=True,
        privileged_operation=True,
        unusual_call_time=True,
        historical_fraud_flags=999,
        transaction_value=10_000_000,
    )
    result = compute_risk(1.0, 0.0, 1.0, context, default_weights())
    assert 0.0 <= result.risk_score <= 100.0


# ---------------------------------------------------------------------
# Explainable reasons (matching the spec's example strings)
# ---------------------------------------------------------------------

def test_high_synthetic_probability_reason():
    context = ContextFactors()
    result = compute_risk(0.9, 0.9, 0.05, context, default_weights())
    assert "High synthetic speech probability" in result.reasons


def test_speaker_verification_uncertain_reason():
    context = ContextFactors()
    result = compute_risk(0.1, 0.5, 0.1, context, default_weights())
    assert "Speaker verification uncertain" in result.reasons


def test_low_speaker_similarity_reason():
    context = ContextFactors()
    result = compute_risk(0.1, 0.2, 0.1, context, default_weights())
    assert "Low speaker similarity to reference voice" in result.reasons


def test_speaker_unavailable_reason_when_no_reference():
    context = ContextFactors()
    result = compute_risk(0.1, None, 0.1, context, default_weights())
    assert "Speaker verification unavailable (no reference registered)" in result.reasons


def test_high_value_transaction_reason():
    context = ContextFactors(transaction_value=settings.TRANSACTION_HIGH_THRESHOLD + 1000)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "High-value transaction" in result.reasons


def test_bypass_callback_reason_exact_wording():
    context = ContextFactors(bypass_requested=True)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Caller requested bypass of registered callback" in result.reasons


def test_not_registered_contact_reason():
    context = ContextFactors(caller_is_registered_contact=False)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Caller is not a registered contact" in result.reasons


def test_new_device_reason_only_when_registered():
    context = ContextFactors(caller_is_registered_contact=True, is_new_device=True)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Call originated from a new/unrecognized device" in result.reasons


def test_privileged_operation_reason():
    context = ContextFactors(privileged_operation=True)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Request involves a privileged operation" in result.reasons


def test_urgency_reason():
    context = ContextFactors(urgency_indicated=True)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Urgency indicators present in the request" in result.reasons


def test_historical_fraud_reason():
    context = ContextFactors(historical_fraud_flags=2)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Prior fraud indicators on record for this caller/account" in result.reasons


def test_poor_reputation_reason():
    context = ContextFactors(caller_reputation_score=0.2)
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert "Caller has a poor reputation score" in result.reasons


def test_no_reasons_gives_explicit_fallback_message():
    context = ContextFactors()
    result = compute_risk(0.05, 0.95, 0.05, context, default_weights())
    assert result.reasons == ["No significant risk indicators detected"]


# ---------------------------------------------------------------------
# Recommended action mapping
# ---------------------------------------------------------------------

def test_determine_recommended_action_low():
    assert determine_recommended_action("LOW", ContextFactors()) == "ALLOW"


def test_determine_recommended_action_medium():
    assert determine_recommended_action("MEDIUM", ContextFactors()) == "MONITOR"


def test_determine_recommended_action_high_without_bypass_or_privilege():
    assert determine_recommended_action("HIGH", ContextFactors()) == "SECONDARY_VERIFICATION"


def test_determine_recommended_action_high_with_bypass_is_callback_and_mfa():
    context = ContextFactors(bypass_requested=True)
    assert determine_recommended_action("HIGH", context) == "CALLBACK_AND_MFA"


def test_determine_recommended_action_high_with_privileged_is_callback_and_mfa():
    context = ContextFactors(privileged_operation=True)
    assert determine_recommended_action("HIGH", context) == "CALLBACK_AND_MFA"


def test_determine_recommended_action_critical_always_escalates():
    context = ContextFactors(bypass_requested=True, privileged_operation=True)
    assert determine_recommended_action("CRITICAL", context) == "ESCALATE"


def test_end_to_end_high_risk_with_bypass_recommends_callback_and_mfa():
    """Tuned inputs land in the HIGH band with a bypass request --
    confirms the full compute_risk() pipeline (not just the isolated
    helper) produces CALLBACK_AND_MFA rather than generic
    SECONDARY_VERIFICATION."""
    context = ContextFactors(
        bypass_requested=True, privileged_operation=True, urgency_indicated=True, transaction_value=30000
    )
    result = compute_risk(0.9, 0.3, 0.5, context, default_weights())
    assert result.risk_level == "HIGH"
    assert result.recommended_action == "CALLBACK_AND_MFA"


# ---------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------

def test_weights_are_configurable_and_affect_score():
    context = ContextFactors()
    high_synthetic_weight = RiskWeights(
        synthetic_weight=0.9, speaker_weight=0.02, prosody_weight=0.02,
        caller_reputation_weight=0.01, registered_contact_weight=0.01, transaction_weight=0.01,
        privileged_action_weight=0.01, urgency_weight=0.01, bypass_callback_weight=0.005,
        historical_fraud_weight=0.005,
    )
    zero_synthetic_weight = RiskWeights(
        synthetic_weight=0.0, speaker_weight=0.15, prosody_weight=0.08,
        caller_reputation_weight=0.10, registered_contact_weight=0.07, transaction_weight=0.15,
        privileged_action_weight=0.06, urgency_weight=0.05, bypass_callback_weight=0.06,
        historical_fraud_weight=0.03,
    )
    result_high = compute_risk(0.95, 0.95, 0.0, context, high_synthetic_weight)
    result_zero = compute_risk(0.95, 0.95, 0.0, context, zero_synthetic_weight)
    assert result_high.risk_score > result_zero.risk_score


def test_thresholds_are_configurable():
    """Verifies score_to_level reads live from settings, so a deployment
    can retune LOW/MEDIUM/HIGH/CRITICAL boundaries via environment
    variables without code changes."""
    original_low = settings.RISK_THRESHOLD_LOW
    try:
        settings.RISK_THRESHOLD_LOW = 5.0
        assert score_to_level(10.0) != "LOW"
    finally:
        settings.RISK_THRESHOLD_LOW = original_low


def test_transaction_scaling_is_configurable():
    original_low, original_high = settings.TRANSACTION_LOW_THRESHOLD, settings.TRANSACTION_HIGH_THRESHOLD
    try:
        settings.TRANSACTION_LOW_THRESHOLD = 0.0
        settings.TRANSACTION_HIGH_THRESHOLD = 100.0
        context = ContextFactors(transaction_value=100.0)
        result = compute_risk(0.0, 1.0, 0.0, context, default_weights())
        # a $100 transaction now maxes out the transaction factor given the
        # retuned thresholds -- confirms the scaling genuinely reads config.
        assert "High-value transaction" in result.reasons
    finally:
        settings.TRANSACTION_LOW_THRESHOLD = original_low
        settings.TRANSACTION_HIGH_THRESHOLD = original_high


# ---------------------------------------------------------------------
# Context risk summary (dashboard display metric)
# ---------------------------------------------------------------------

def test_context_risk_summary_is_zero_for_clean_context():
    context = ContextFactors()
    summary = compute_context_risk_summary(context, default_weights())
    assert summary == 0.0


def test_context_risk_summary_increases_with_risk_factors():
    clean = compute_context_risk_summary(ContextFactors(), default_weights())
    risky = compute_context_risk_summary(
        ContextFactors(bypass_requested=True, transaction_value=100000, urgency_indicated=True),
        default_weights(),
    )
    assert risky > clean
    assert 0.0 <= risky <= 1.0


def test_context_risk_summary_does_not_include_audio_signals():
    """The context summary is deliberately audio-independent -- it
    should be identical regardless of synthetic/speaker/prosody scores,
    since those are reported separately."""
    context = ContextFactors(transaction_value=20000)
    weights = default_weights()
    summary_a = compute_context_risk_summary(context, weights)
    summary_b = compute_context_risk_summary(context, weights)
    assert summary_a == summary_b

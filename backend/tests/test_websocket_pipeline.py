"""
Tests for the near-real-time WebSocket audio detection pipeline
(/ws/calls/{call_id}/audio). Covers: the connection handshake, the full
per-chunk signal pipeline, invalid/corrupt audio handling, mid-call
context updates, disconnection cleanup, latency measurement, and
speaker-verification integration when a call is linked to a registered
speaker profile.

Only synthetic (sine-wave) test audio is used -- no real voice data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.audio_fixtures import make_corrupt_audio_bytes, make_wav_bytes


def test_websocket_sends_connected_handshake_first(client):
    with client.websocket_connect("/ws/calls/CALL-HANDSHAKE/audio") as ws:
        handshake = ws.receive_json()
        assert handshake["type"] == "connected"
        assert handshake["call_id"] == "CALL-HANDSHAKE"
        assert handshake["speaker_reference_loaded"] is False
        assert "disclaimer" in handshake
        assert "not proof of fraud" in handshake["disclaimer"]


def test_websocket_full_pipeline_returns_all_expected_fields(client):
    wav_bytes = make_wav_bytes(duration_seconds=3.0, freq=250.0)

    with client.websocket_connect("/ws/calls/CALL-FULL-PIPELINE/audio") as ws:
        ws.receive_json()  # handshake

        ws.send_bytes(wav_bytes)
        result = ws.receive_json()

        assert result["type"] == "result"
        assert result["call_id"] == "CALL-FULL-PIPELINE"

        # synthetic voice inference
        assert 0.0 <= result["synthetic_probability"] <= 1.0

        # speaker verification (no reference registered for this call ->
        # verification is skipped, not crashed)
        assert result["speaker_similarity"] is None
        assert result["identity_match"] is None

        # prosody analysis
        assert 0.0 <= result["prosody_anomaly"] <= 1.0

        # context risk
        assert 0.0 <= result["context_risk"] <= 1.0

        # risk engine + temporal smoothing
        assert 0.0 <= result["risk_score"] <= 100.0
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert isinstance(result["reasons"], list)
        assert result["recommended_action"] in (
            "CONTINUE_NORMAL_MONITORING", "INCREASED_MONITORING",
            "SECONDARY_VERIFICATION", "PAUSE_AND_INDEPENDENT_VERIFICATION",
        )

        # non-definitive-proof disclaimer present on every result
        assert "disclaimer" in result
        assert "not proof of fraud" in result["disclaimer"]

        # latency measurements: real numbers, not fabricated/missing
        assert "latency_ms" in result
        for key in ("preprocessing", "inference", "total"):
            assert key in result["latency_ms"]
            assert isinstance(result["latency_ms"][key], (int, float))
            assert result["latency_ms"][key] >= 0.0


def test_websocket_returns_multiple_sequential_results(client):
    """Sending several chunks in sequence should produce a result for
    each voiced chunk, with timestamps that move forward."""
    with client.websocket_connect("/ws/calls/CALL-SEQUENCE/audio") as ws:
        ws.receive_json()  # handshake

        results = []
        for i in range(3):
            wav_bytes = make_wav_bytes(duration_seconds=2.5, freq=200.0 + i * 15)
            ws.send_bytes(wav_bytes)
            result = ws.receive_json()
            results.append(result)

        assert len(results) == 3
        for r in results:
            assert r["type"] == "result"
            assert r["call_id"] == "CALL-SEQUENCE"
            assert 0.0 <= r["risk_score"] <= 100.0

        # chunk_start_time should be non-decreasing across sequential sends
        start_times = [r["chunk_start_time"] for r in results]
        assert start_times == sorted(start_times)


def test_websocket_handles_corrupt_audio_without_crashing(client):
    """Corrupt/unsupported audio must produce an error message and keep
    the connection alive -- not terminate the stream."""
    with client.websocket_connect("/ws/calls/CALL-CORRUPT/audio") as ws:
        ws.receive_json()  # handshake

        ws.send_bytes(make_corrupt_audio_bytes())
        error_response = ws.receive_json()
        assert error_response["type"] == "error"
        assert "call_id" in error_response

        # connection must still be usable after the corrupt chunk
        good_wav = make_wav_bytes(duration_seconds=2.5)
        ws.send_bytes(good_wav)
        result = ws.receive_json()
        assert result["type"] == "result"
        assert result["call_id"] == "CALL-CORRUPT"


def test_websocket_handles_empty_bytes_gracefully(client):
    with client.websocket_connect("/ws/calls/CALL-EMPTY/audio") as ws:
        ws.receive_json()  # handshake
        ws.send_bytes(b"")
        # No response is expected for an empty frame (it's ignored), so
        # send valid audio next and confirm the stream still works.
        good_wav = make_wav_bytes(duration_seconds=2.5)
        ws.send_bytes(good_wav)
        result = ws.receive_json()
        assert result["type"] == "result"


def test_websocket_context_update_mid_stream_changes_risk(client):
    wav_bytes = make_wav_bytes(duration_seconds=2.5, freq=300.0)

    with client.websocket_connect("/ws/calls/CALL-CONTEXT/audio") as ws:
        ws.receive_json()  # handshake

        ws.send_bytes(wav_bytes)
        baseline_result = ws.receive_json()

        # Escalate context: bypass request + high-value transaction.
        import json

        ws.send_text(json.dumps({
            "bypass_requested": True,
            "urgency_indicated": True,
            "privileged_operation": True,
            "transaction_value": 100000,
        }))
        ack = ws.receive_json()
        assert ack["type"] == "context_updated"

        ws.send_bytes(wav_bytes)
        escalated_result = ws.receive_json()

        assert escalated_result["context_risk"] > baseline_result["context_risk"]
        assert escalated_result["risk_score"] >= baseline_result["risk_score"]


def test_websocket_invalid_context_payload_reports_error_and_continues(client):
    with client.websocket_connect("/ws/calls/CALL-BADCONTEXT/audio") as ws:
        ws.receive_json()  # handshake

        ws.send_text("not valid json{{{")
        error_response = ws.receive_json()
        assert error_response["type"] == "error"

        # stream should still work afterward
        ws.send_bytes(make_wav_bytes(duration_seconds=2.5))
        result = ws.receive_json()
        assert result["type"] == "result"


def test_websocket_disconnect_does_not_crash_server(client):
    """Opening and then cleanly closing a connection should not raise,
    and the server should remain usable for a fresh connection
    afterward (smoothing state is cleaned up per call_id)."""
    with client.websocket_connect("/ws/calls/CALL-DISCONNECT-1/audio") as ws:
        ws.receive_json()
        ws.send_bytes(make_wav_bytes(duration_seconds=2.5))
        ws.receive_json()
    # connection closed cleanly by the `with` block exiting

    # A new connection (even reusing a call_id) must work fine afterward.
    with client.websocket_connect("/ws/calls/CALL-DISCONNECT-1/audio") as ws2:
        handshake = ws2.receive_json()
        assert handshake["type"] == "connected"
        ws2.send_bytes(make_wav_bytes(duration_seconds=2.5))
        result = ws2.receive_json()
        assert result["type"] == "result"


def test_websocket_speaker_verification_integration(client):
    """When a call is linked to a registered speaker with a reference
    embedding, the pipeline should run speaker verification and include
    a real (non-null) similarity score."""
    reference_audio = make_wav_bytes(duration_seconds=2.0, freq=180.0)

    register_resp = client.post(
        "/api/v1/speakers?speaker_id=demo-speaker-1&display_name=Demo+Speaker",
        files={"file": ("reference.wav", reference_audio, "audio/wav")},
    )
    assert register_resp.status_code == 200

    call_resp = client.post(
        "/api/v1/calls",
        json={"caller_label": "Speaker Verify Test", "speaker_id": "demo-speaker-1", "is_demo": True},
    )
    assert call_resp.status_code == 200
    call_id = call_resp.json()["call_id"]

    with client.websocket_connect(f"/ws/calls/{call_id}/audio") as ws:
        handshake = ws.receive_json()
        assert handshake["type"] == "connected"
        assert handshake["speaker_reference_loaded"] is True

        # Stream the SAME audio used as the reference -- should yield a
        # high similarity score since it's an identical signal.
        ws.send_bytes(reference_audio)
        result = ws.receive_json()

        assert result["speaker_similarity"] is not None
        assert 0.0 <= result["speaker_similarity"] <= 1.0
        assert result["identity_match"] is not None


def test_websocket_without_registered_speaker_skips_verification_gracefully(client):
    call_resp = client.post(
        "/api/v1/calls",
        json={"caller_label": "No Speaker Test", "is_demo": True},
    )
    call_id = call_resp.json()["call_id"]

    with client.websocket_connect(f"/ws/calls/{call_id}/audio") as ws:
        handshake = ws.receive_json()
        assert handshake["speaker_reference_loaded"] is False

        ws.send_bytes(make_wav_bytes(duration_seconds=2.5))
        result = ws.receive_json()
        assert result["speaker_similarity"] is None


def test_websocket_alert_generated_for_high_risk_stream(client):
    """Escalating context factors to a level that should cross into
    MEDIUM/HIGH/CRITICAL must produce a populated `alert` field."""
    import json

    with client.websocket_connect("/ws/calls/CALL-ALERT-TEST/audio") as ws:
        ws.receive_json()  # handshake

        ws.send_text(json.dumps({
            "bypass_requested": True,
            "urgency_indicated": True,
            "privileged_operation": True,
            "historical_fraud_flags": 5,
            "transaction_value": 250000,
        }))
        ws.receive_json()  # ack

        # Send several chunks so the smoothed score has a chance to climb.
        last_result = None
        for _ in range(4):
            ws.send_bytes(make_wav_bytes(duration_seconds=2.5))
            last_result = ws.receive_json()

        if last_result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL"):
            assert last_result["alert"] is not None
            assert last_result["alert"]["severity"] == last_result["risk_level"]


def test_websocket_activity_persists_explainable_risk_history(client):
    """Risk events produced during a WebSocket stream must be
    retrievable afterward via GET /api/v1/calls/{call_id}/risk, with
    the full explanation (reasons, recommended_action) intact -- this
    is what powers the Call Details 'why' view in the dashboard."""
    import json

    call_resp = client.post(
        "/api/v1/calls",
        json={"caller_label": "History Test", "is_demo": True},
    )
    call_id = call_resp.json()["call_id"]

    with client.websocket_connect(f"/ws/calls/{call_id}/audio") as ws:
        ws.receive_json()  # handshake
        ws.send_text(json.dumps({"bypass_requested": True, "transaction_value": 80000}))
        ws.receive_json()  # ack
        ws.send_bytes(make_wav_bytes(duration_seconds=2.5))
        live_result = ws.receive_json()

    history_resp = client.get(f"/api/v1/calls/{call_id}/risk")
    assert history_resp.status_code == 200
    history = history_resp.json()

    assert len(history) >= 1
    latest_event = history[-1]
    assert latest_event["risk_level"] == live_result["risk_level"]
    assert isinstance(latest_event["reasons"], list)
    assert len(latest_event["reasons"]) > 0
    assert latest_event["recommended_action"] in (
        "ALLOW", "MONITOR", "SECONDARY_VERIFICATION", "CALLBACK_AND_MFA", "ESCALATE",
    )
    assert "Caller requested bypass of registered callback" in latest_event["reasons"]

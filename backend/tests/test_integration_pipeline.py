import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.audio_fixtures import make_wav_bytes as _make_wav_bytes


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["app"] == "VOICEGUARD"


def test_create_and_get_call(client):
    resp = client.post("/api/v1/calls", json={"caller_label": "Test Caller", "is_demo": True})
    assert resp.status_code == 200
    call = resp.json()
    assert call["call_id"].startswith("CALL-")

    resp2 = client.get(f"/api/v1/calls/{call['call_id']}")
    assert resp2.status_code == 200
    assert resp2.json()["call_id"] == call["call_id"]


def test_upload_detection_to_risk_flow(client):
    wav_bytes = _make_wav_bytes()

    detect_resp = client.post(
        "/api/v1/detection/analyze",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert detect_resp.status_code == 200
    detection = detect_resp.json()
    assert "average_synthetic_probability" in detection
    assert 0.0 <= detection["average_synthetic_probability"] <= 1.0

    risk_resp = client.post(
        "/api/v1/risk/evaluate",
        params={
            "synthetic_probability": detection["average_synthetic_probability"],
            "speaker_similarity": 0.4,
            "prosody_anomaly": detection["average_prosody_anomaly"],
        },
        json={"bypass_requested": True, "transaction_value": 50000},
    )
    assert risk_resp.status_code == 200
    risk = risk_resp.json()
    assert risk["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    if risk["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL"):
        alert_resp = client.post(
            "/api/v1/alerts",
            json={
                "call_id": "CALL-TEST0001",
                "severity": risk["risk_level"],
                "message": "test alert",
                "recommended_action": risk["recommended_action"],
            },
        )
        assert alert_resp.status_code == 200


def test_rejects_unsupported_file_type(client):
    resp = client.post(
        "/api/v1/detection/analyze",
        files={"file": ("sample.exe", b"not audio", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_websocket_call_stream_roundtrip(client):
    wav_bytes = _make_wav_bytes(duration_seconds=3.0)

    with client.websocket_connect("/ws/calls/CALL-WS-TEST/audio") as ws:
        handshake = ws.receive_json()
        assert handshake["type"] == "connected"
        assert handshake["call_id"] == "CALL-WS-TEST"

        ws.send_bytes(wav_bytes)
        response = ws.receive_json()
        assert response["type"] == "result"
        assert "risk_score" in response
        assert response["call_id"] == "CALL-WS-TEST"
        assert response["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

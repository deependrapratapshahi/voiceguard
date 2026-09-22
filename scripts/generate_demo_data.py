"""
Generates synthetic (non-audio) demo metadata for exercising the API
and dashboard without needing real audio files or telecom integration:
demo calls, risk-event history, and alerts across the three documented
scenarios (genuine caller / synthetic caller / high-risk impersonation).

This script talks to the running backend over HTTP -- start the backend
first (`make backend` or `docker compose up backend`), then run:

    python scripts/generate_demo_data.py

It does not generate or clone any real person's voice; it only creates
numeric risk/alert records via the REST API, matching the shapes the
detection/prosody pipeline would normally produce from real audio.
"""
from __future__ import annotations

import random
import time

import httpx

BASE_URL = "http://localhost:8000/api/v1"

SCENARIOS = [
    {
        "caller_label": "Genuine Caller (Demo)",
        "synthetic_curve": [0.05, 0.08, 0.06, 0.07],
        "speaker_similarity": 0.91,
        "prosody_anomaly": 0.10,
        "context": {"bypass_requested": False, "urgency_indicated": False, "transaction_value": 200},
    },
    {
        "caller_label": "Synthetic Caller (Demo)",
        "synthetic_curve": [0.55, 0.68, 0.74, 0.81],
        "speaker_similarity": 0.62,
        "prosody_anomaly": 0.55,
        "context": {"bypass_requested": False, "urgency_indicated": True, "transaction_value": 5000},
    },
    {
        "caller_label": "High-Risk Impersonation (Demo)",
        "synthetic_curve": [0.65, 0.78, 0.85, 0.92],
        "speaker_similarity": 0.31,
        "prosody_anomaly": 0.71,
        "context": {
            "bypass_requested": True,
            "urgency_indicated": True,
            "privileged_operation": True,
            "transaction_value": 75000,
        },
    },
]


def main():
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)

    try:
        client.get("/health").raise_for_status()
    except Exception as exc:
        print(f"Backend not reachable at {BASE_URL}: {exc}")
        print("Start it first with `make backend` or `docker compose up backend`.")
        return

    for scenario in SCENARIOS:
        call = client.post(
            "/calls", json={"caller_label": scenario["caller_label"], "is_demo": True}
        ).json()
        call_id = call["call_id"]
        print(f"Created {call_id} for scenario '{scenario['caller_label']}'")

        for synthetic_prob in scenario["synthetic_curve"]:
            risk = client.post(
                "/risk/evaluate",
                params={
                    "synthetic_probability": synthetic_prob,
                    "speaker_similarity": scenario["speaker_similarity"],
                    "prosody_anomaly": scenario["prosody_anomaly"],
                },
                json=scenario["context"],
            ).json()
            print(f"  risk={risk['risk_score']} level={risk['risk_level']}")

            if risk["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL"):
                client.post(
                    "/alerts",
                    json={
                        "call_id": call_id,
                        "severity": risk["risk_level"],
                        "message": "; ".join(risk["reasons"]),
                        "recommended_action": risk["recommended_action"],
                    },
                )
            time.sleep(0.2)

    print("\nDemo data generation complete. Open the dashboard to view results.")


if __name__ == "__main__":
    main()

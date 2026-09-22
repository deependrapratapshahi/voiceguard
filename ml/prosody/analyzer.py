"""
Converts raw prosodic features into a bounded "prosody anomaly" score.

Baseline approach: compares extracted features against configurable
expected ranges for natural human speech and produces a 0-1 anomaly
score. This is explicitly supporting evidence, not proof of synthetic
speech, and is combined with other signals downstream in the risk engine.
"""
from __future__ import annotations

import numpy as np

from ml.prosody.features import extract_prosody_features

# Rough expected ranges for natural conversational speech; deliberately
# conservative and configurable so they can be tuned per deployment/locale.
EXPECTED_RANGES = {
    "pitch_std_hz": (10.0, 60.0),
    "jitter": (0.0, 0.05),
    "shimmer": (0.0, 0.08),
    "pause_frequency": (0.05, 0.45),
}


def _range_penalty(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 0.0
    span = max(high - low, 1e-6)
    if value < low:
        return min(1.0, (low - value) / span)
    return min(1.0, (value - high) / span)


def compute_prosody_anomaly(samples: np.ndarray, sr: int) -> tuple[float, dict]:
    features = extract_prosody_features(samples, sr)

    penalties = []
    for key, (low, high) in EXPECTED_RANGES.items():
        penalties.append(_range_penalty(features.get(key, 0.0), low, high))

    anomaly_score = float(np.clip(np.mean(penalties), 0.0, 1.0)) if penalties else 0.0
    return anomaly_score, features

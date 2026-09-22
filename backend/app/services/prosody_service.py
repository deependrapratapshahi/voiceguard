from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(ML_ROOT.parent) not in sys.path:
    sys.path.append(str(ML_ROOT.parent))

from ml.prosody.analyzer import compute_prosody_anomaly  # noqa: E402

from app.models.schemas import ProsodyResult
from app.services.audio_service import AudioChunk


def analyze_chunk(chunk: AudioChunk) -> ProsodyResult:
    anomaly, features = compute_prosody_anomaly(chunk.samples, chunk.sample_rate)
    return ProsodyResult(prosody_anomaly=anomaly, features=features)

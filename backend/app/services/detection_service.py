"""
Detection service: bridges the FastAPI layer and the ml/ package.

Kept intentionally thin -- all actual signal processing and modeling
logic lives in app/services/audio_service.py and ml/deepfake/*.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

# Allow importing the sibling `ml/` package from the backend app.
ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(ML_ROOT.parent) not in sys.path:
    sys.path.append(str(ML_ROOT.parent))

from ml.deepfake.inference import run_detection  # noqa: E402

from app.config import get_settings
from app.models.schemas import DetectionResult
from app.services.audio_service import AudioChunk

settings = get_settings()


def analyze_chunk(chunk: AudioChunk) -> DetectionResult:
    result = run_detection(
        samples=chunk.samples,
        sr=chunk.sample_rate,
        model_dir=settings.MODEL_PATH,
        use_pretrained=settings.USE_PRETRAINED_ENCODERS,
    )
    return DetectionResult(
        synthetic_probability=result.synthetic_probability,
        label=result.label,
        model_version=result.model_version,
    )


def analyze_chunks(chunks: List[AudioChunk]) -> List[DetectionResult]:
    return [analyze_chunk(c) for c in chunks if c.is_voiced] or [
        analyze_chunk(c) for c in chunks
    ]

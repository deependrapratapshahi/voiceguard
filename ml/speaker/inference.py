"""
Speaker inference entry point used by the backend speaker_service.
"""
from __future__ import annotations

import numpy as np

from ml.speaker.embeddings import extract_baseline_embedding
from ml.speaker.verification import VerificationResult, verify


def embed(samples: np.ndarray, sr: int) -> np.ndarray:
    # Pretrained-encoder path would be selected here, mirroring
    # ml/deepfake/inference.py's fallback pattern, once real encoder
    # weights are wired in.
    return extract_baseline_embedding(samples, sr)


def verify_against_reference(
    reference_embedding: list[float], samples: np.ndarray, sr: int, threshold: float
) -> VerificationResult:
    candidate = embed(samples, sr)
    reference = np.array(reference_embedding, dtype=np.float32)
    return verify(reference, candidate, threshold=threshold)

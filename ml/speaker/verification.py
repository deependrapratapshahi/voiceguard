"""
Speaker verification: compares an incoming-call embedding against a
stored reference embedding using cosine similarity.

Important: a given similarity score does NOT universally mean "same
person" -- the decision threshold is configurable per deployment and
should be calibrated against a representative dataset (see
`calibrate_threshold`). Treat `identity_match` as a configurable
decision, not a ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SIMILARITY_THRESHOLD = 0.55


@dataclass
class VerificationResult:
    speaker_similarity: float
    identity_match: bool
    confidence: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    # Map cosine similarity [-1, 1] to [0, 1] for a probability-like score.
    raw = float(np.dot(a, b) / (a_norm * b_norm))
    return (raw + 1.0) / 2.0


def verify(reference_embedding: np.ndarray, candidate_embedding: np.ndarray,
           threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> VerificationResult:
    similarity = cosine_similarity(reference_embedding, candidate_embedding)
    match = similarity >= threshold
    # Confidence is expressed as distance from the threshold, not as a
    # claim of certainty.
    confidence = float(min(1.0, abs(similarity - threshold) / max(threshold, 1e-6) + 0.5))
    return VerificationResult(speaker_similarity=similarity, identity_match=match, confidence=confidence)


def calibrate_threshold(genuine_scores: list[float], impostor_scores: list[float]) -> float:
    """
    Simple EER-style calibration helper: scans candidate thresholds and
    returns the one minimizing the difference between false-accept and
    false-reject rates on provided labeled score sets. Intended for use
    with a held-out calibration dataset, not production traffic.
    """
    if not genuine_scores or not impostor_scores:
        return DEFAULT_SIMILARITY_THRESHOLD

    candidates = sorted(set(genuine_scores + impostor_scores))
    best_threshold = DEFAULT_SIMILARITY_THRESHOLD
    best_gap = float("inf")

    for t in candidates:
        far = sum(1 for s in impostor_scores if s >= t) / len(impostor_scores)
        frr = sum(1 for s in genuine_scores if s < t) / len(genuine_scores)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_threshold = t

    return best_threshold

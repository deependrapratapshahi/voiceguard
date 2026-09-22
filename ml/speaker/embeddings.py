"""
Speaker embedding extraction.

Baseline: a simple, deterministic embedding built from summarized MFCC +
pitch statistics (classical, no training required). This is clearly a
weak baseline for a real speaker-verification product, and is designed
to be swapped for a pretrained embedding model (e.g. ECAPA-TDNN, or a
Resemblyzer/SpeechBrain speaker encoder) without changing the interface
used by verification.py.
"""
from __future__ import annotations

import numpy as np
import librosa

EMBEDDING_MODEL_VERSION = "baseline-v1"


def extract_baseline_embedding(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(46, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    f0, voiced_flag, _ = librosa.pyin(
        samples, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(f0_valid)) if f0_valid.size else 0.0
    pitch_std = float(np.std(f0_valid)) if f0_valid.size else 0.0

    embedding = np.concatenate(
        [mfcc_mean, mfcc_std, np.array([pitch_mean, pitch_std, EMBEDDING_MODEL_VERSION.count("-")])]
    )
    norm = np.linalg.norm(embedding)
    return (embedding / norm).astype(np.float32) if norm > 1e-8 else embedding.astype(np.float32)


class PretrainedSpeakerEncoder:
    """
    Optional interface for a pretrained speaker-embedding model
    (e.g. ECAPA-TDNN via SpeechBrain, or a Hugging Face speaker encoder).
    Only used if weights are available locally; otherwise the baseline
    embedding above is used (see ml/speaker/inference.py).
    """

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.exists()

    def extract(self, samples: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError(
            "PretrainedSpeakerEncoder.extract must be wired to real model "
            "weights before use; falls back to the baseline embedding until then."
        )

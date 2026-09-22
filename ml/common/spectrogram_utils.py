"""
Pure-numpy spectrogram post-processing utilities.

Deliberately has NO dependency on librosa/soundfile/torch so this logic
(padding, truncation, normalization) can be unit-tested in any
environment, including ones where the heavier audio/ML libraries are
not installed.
"""
from __future__ import annotations

import numpy as np


def pad_or_truncate_and_normalize(
    log_mel: np.ndarray, fixed_frames: int = 128, n_mels: int = 64
) -> np.ndarray:
    """
    Pads/truncates a (n_mels, time) log-mel matrix to exactly
    `fixed_frames` time steps and applies per-instance normalization
    (zero mean, unit variance) since raw log-mel values otherwise vary
    a lot with recording loudness, which destabilizes CNN training.
    """
    if log_mel.size == 0 or log_mel.ndim != 2 or log_mel.shape[1] == 0:
        return np.zeros((n_mels, fixed_frames), dtype=np.float32)

    n_frames = log_mel.shape[1]
    if n_frames < fixed_frames:
        pad_width = fixed_frames - n_frames
        pad_value = float(np.min(log_mel))
        log_mel = np.pad(
            log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_value
        )
    elif n_frames > fixed_frames:
        log_mel = log_mel[:, :fixed_frames]

    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6
    normalized = (log_mel - mean) / std
    return normalized.astype(np.float32)

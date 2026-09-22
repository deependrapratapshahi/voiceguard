"""
Shared feature-extraction helpers used by the baseline deepfake, speaker,
and prosody models. Kept dependency-light (numpy/librosa/scipy only) so
the baseline path works without GPU or heavyweight transformer downloads.
"""
from __future__ import annotations

import numpy as np
import librosa

from ml.common.spectrogram_utils import pad_or_truncate_and_normalize


def extract_log_mel(
    samples: np.ndarray,
    sr: int,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # shape: (n_mels, time)


def extract_log_mel_fixed(
    samples: np.ndarray,
    sr: int,
    n_mels: int = 64,
    fixed_frames: int = 128,
    hop_length: int = 256,
    n_fft: int = 1024,
) -> np.ndarray:
    """
    Log-mel spectrogram padded/truncated to a fixed number of time
    frames, for feeding fixed-input-size models (e.g. the CNN baseline
    in ml/deepfake/model.py). See pad_or_truncate_and_normalize() for
    the padding/normalization logic.
    """
    log_mel = extract_log_mel(samples, sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    return pad_or_truncate_and_normalize(log_mel, fixed_frames=fixed_frames, n_mels=n_mels)


def extract_mfcc(samples: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    return librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)


def summarize_feature_matrix(feat: np.ndarray) -> np.ndarray:
    """Collapse a (n_features, time) matrix into a fixed-length vector
    (mean + std per feature) suitable for classical classifiers."""
    if feat.size == 0:
        return np.zeros(0, dtype=np.float32)
    mean = np.mean(feat, axis=1)
    std = np.std(feat, axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_baseline_feature_vector(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Baseline, model-agnostic feature vector: summarized log-mel + MFCC.
    Used by the classical fallback classifier when pretrained speech
    encoders (wav2vec2/HuBERT/WavLM) are unavailable or disabled.
    """
    if samples.size == 0:
        return np.zeros(168, dtype=np.float32)
    log_mel = extract_log_mel(samples, sr)
    mfcc = extract_mfcc(samples, sr)
    return np.concatenate([summarize_feature_matrix(log_mel), summarize_feature_matrix(mfcc)])

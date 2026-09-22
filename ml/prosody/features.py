"""
Prosodic and acoustic feature extraction: pitch/F0, pitch variance,
speech rate proxy, pause statistics, RMS energy, jitter, shimmer, and
basic spectral characteristics.

These are classical DSP features (via librosa/numpy) -- no model
training required, which keeps this component fully functional in
baseline/fallback mode.
"""
from __future__ import annotations

import numpy as np
import librosa


def extract_prosody_features(samples: np.ndarray, sr: int) -> dict:
    if samples.size == 0:
        return _empty_features()

    f0, voiced_flag, _ = librosa.pyin(
        samples, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    rms = librosa.feature.rms(y=samples)[0]
    zero_crossings = librosa.feature.zero_crossing_rate(samples)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]

    pause_frames = rms < (0.1 * np.max(rms)) if rms.size else np.array([])
    pause_frequency = float(np.mean(pause_frames)) if pause_frames.size else 0.0

    jitter, shimmer = _estimate_jitter_shimmer(f0_valid, rms)

    return {
        "pitch_mean_hz": float(np.mean(f0_valid)) if f0_valid.size else 0.0,
        "pitch_std_hz": float(np.std(f0_valid)) if f0_valid.size else 0.0,
        "voiced_ratio": float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else 0.0,
        "speech_rate_proxy": float(np.mean(zero_crossings)) if zero_crossings.size else 0.0,
        "pause_frequency": pause_frequency,
        "rms_energy_mean": float(np.mean(rms)) if rms.size else 0.0,
        "rms_energy_std": float(np.std(rms)) if rms.size else 0.0,
        "jitter": jitter,
        "shimmer": shimmer,
        "spectral_centroid_mean": float(np.mean(spectral_centroid)) if spectral_centroid.size else 0.0,
    }


def _estimate_jitter_shimmer(f0_valid: np.ndarray, rms: np.ndarray) -> tuple[float, float]:
    jitter = 0.0
    if f0_valid.size > 1:
        diffs = np.abs(np.diff(f0_valid))
        jitter = float(np.mean(diffs) / (np.mean(f0_valid) + 1e-8))

    shimmer = 0.0
    if rms.size > 1:
        diffs = np.abs(np.diff(rms))
        shimmer = float(np.mean(diffs) / (np.mean(rms) + 1e-8))

    return jitter, shimmer


def _empty_features() -> dict:
    return {
        "pitch_mean_hz": 0.0, "pitch_std_hz": 0.0, "voiced_ratio": 0.0,
        "speech_rate_proxy": 0.0, "pause_frequency": 0.0, "rms_energy_mean": 0.0,
        "rms_energy_std": 0.0, "jitter": 0.0, "shimmer": 0.0, "spectral_centroid_mean": 0.0,
    }

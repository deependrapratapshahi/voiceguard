"""
Preprocessing for the deepfake training pipeline: loads audio for each
dataset item and extracts features for either the classical baseline
(summarized log-mel + MFCC vector) or the CNN baseline (fixed-size
log-mel spectrogram). Kept separate from inference-time preprocessing
(app/services/audio_service.py) since training can batch-process from
disk rather than streaming buffers.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa

from ml.common.audio import extract_baseline_feature_vector, extract_log_mel_fixed
from ml.common.configuration import DEFAULT_SPECTROGRAM_CONFIG, SpectrogramConfig
from ml.deepfake.dataset import DatasetItem

TARGET_SR = 16000


def _load_normalized_samples(path: str) -> np.ndarray:
    """Decode an audio file to a 16kHz mono, peak-normalized waveform.
    Shared by both the classical and CNN feature-extraction paths so
    train-time preprocessing matches inference-time preprocessing."""
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if sr != TARGET_SR:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1e-8:
        samples = samples / peak
    return samples


# --- Classical baseline (summarized feature vector) --------------------

def load_and_featurize(item: DatasetItem) -> np.ndarray | None:
    try:
        samples = _load_normalized_samples(str(item.path))
        return extract_baseline_feature_vector(samples, TARGET_SR)
    except Exception:
        return None


def build_feature_matrix(items: list[DatasetItem]) -> tuple[np.ndarray, np.ndarray, list[DatasetItem]]:
    """Returns (X, y, kept_items) skipping any files that failed to load."""
    features = []
    labels = []
    kept = []
    for item in items:
        vec = load_and_featurize(item)
        if vec is None:
            continue
        features.append(vec)
        labels.append(item.label)
        kept.append(item)

    if not features:
        return np.zeros((0, 168)), np.zeros(0), []

    return np.vstack(features), np.array(labels), kept


# --- CNN baseline (fixed-size log-mel spectrogram) ----------------------

def load_and_extract_spectrogram(
    item: DatasetItem, config: SpectrogramConfig = DEFAULT_SPECTROGRAM_CONFIG
) -> np.ndarray | None:
    try:
        samples = _load_normalized_samples(str(item.path))
        return extract_log_mel_fixed(
            samples,
            TARGET_SR,
            n_mels=config.n_mels,
            fixed_frames=config.fixed_frames,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
        )
    except Exception:
        return None


def build_spectrogram_matrix(
    items: list[DatasetItem], config: SpectrogramConfig = DEFAULT_SPECTROGRAM_CONFIG
) -> tuple[np.ndarray, np.ndarray, list[DatasetItem]]:
    """
    Returns (X, y, kept_items) where X has shape
    (n_kept, 1, n_mels, fixed_frames) -- ready to feed to
    ml.deepfake.model.SyntheticVoiceCNN -- and files that failed to
    load are skipped (not silently zero-filled, so a corrupt file
    cannot masquerade as a valid all-silence sample).
    """
    spectrograms = []
    labels = []
    kept = []
    for item in items:
        spec = load_and_extract_spectrogram(item, config)
        if spec is None:
            continue
        spectrograms.append(spec)
        labels.append(item.label)
        kept.append(item)

    if not spectrograms:
        empty_shape = (0, 1, config.n_mels, config.fixed_frames)
        return np.zeros(empty_shape, dtype=np.float32), np.zeros(0, dtype=np.float32), []

    X = np.stack(spectrograms)[:, np.newaxis, :, :].astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, kept

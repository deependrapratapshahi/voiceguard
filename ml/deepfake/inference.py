"""
Inference entry point for the synthetic-voice detector.

This is the ONLY function the backend service layer should call
(app/services/detection_service.py). It selects a model in this order:

  1. Optional pretrained self-supervised encoder, if
     `use_pretrained=True` AND weights are found locally.
  2. The CNN baseline (SyntheticVoiceCNN), if a trained checkpoint
     exists at `<model_dir>/deepfake_cnn_baseline.pt`.
  3. If no CNN checkpoint exists yet, an explicit UNCERTAIN result
     labeled "cnn-baseline-v1-untrained" -- never a fabricated
     confident score.
  4. If PyTorch itself is unavailable at runtime, the classical
     scikit-learn fallback detector (trained or untrained).

No exception from an optional path is allowed to crash the caller --
every branch is wrapped so a missing dependency or a corrupt checkpoint
degrades to the next fallback instead of taking down the API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ml.common.audio import extract_baseline_feature_vector, extract_log_mel_fixed
from ml.common.configuration import (
    CNN_CHECKPOINT_FILENAME,
    CLASSICAL_CHECKPOINT_FILENAME,
    CNN_MODEL_VERSION,
    SpectrogramConfig,
)
from ml.deepfake.model import BaselineClassicalDetector, DetectorOutput, PretrainedEncoderDetector

try:
    import torch

    from ml.deepfake.model import load_checkpoint

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_classical_singleton: Optional[BaselineClassicalDetector] = None
_cnn_singleton = None  # (model, SpectrogramConfig) tuple, cached after first load
_cnn_checkpoint_path_loaded: Optional[str] = None


def _get_classical_baseline(model_dir: Path) -> BaselineClassicalDetector:
    global _classical_singleton
    if _classical_singleton is None:
        weights_path = model_dir / CLASSICAL_CHECKPOINT_FILENAME
        _classical_singleton = BaselineClassicalDetector(weights_path)
    return _classical_singleton


def _get_cnn_model(model_dir: Path):
    """Returns (model, SpectrogramConfig) or None if no checkpoint exists.
    Caches the loaded model in-process; if the checkpoint file changes
    (e.g. a new training run), reload it."""
    global _cnn_singleton, _cnn_checkpoint_path_loaded

    checkpoint_path = model_dir / CNN_CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None

    checkpoint_key = str(checkpoint_path.resolve())
    if _cnn_singleton is not None and _cnn_checkpoint_path_loaded == checkpoint_key:
        return _cnn_singleton

    model, config_dict = load_checkpoint(checkpoint_path)
    config = SpectrogramConfig.from_dict(config_dict)
    _cnn_singleton = (model, config, config_dict.get("model_version", CNN_MODEL_VERSION))
    _cnn_checkpoint_path_loaded = checkpoint_key
    return _cnn_singleton


def _run_cnn_detection(samples: np.ndarray, sr: int, model_dir: Path) -> Optional[DetectorOutput]:
    """Returns a DetectorOutput if the CNN path is usable, or None to
    signal the caller should fall back (no checkpoint yet, or an
    unexpected runtime failure)."""
    cnn_state = _get_cnn_model(model_dir)
    if cnn_state is None:
        return DetectorOutput(
            synthetic_probability=0.5,
            label="UNCERTAIN",
            model_version=f"{CNN_MODEL_VERSION}-untrained",
        )

    model, config, model_version = cnn_state
    spectrogram = extract_log_mel_fixed(
        samples,
        sr,
        n_mels=config.n_mels,
        fixed_frames=config.fixed_frames,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
    )
    tensor = torch.from_numpy(spectrogram[np.newaxis, np.newaxis, :, :].astype(np.float32))

    model.eval()
    with torch.no_grad():
        logit = model(tensor)
        probability = float(torch.sigmoid(logit).item())

    from ml.deepfake.model import probability_to_label

    return DetectorOutput(
        synthetic_probability=probability,
        label=probability_to_label(probability),
        model_version=model_version,
    )


def run_detection(samples: np.ndarray, sr: int, model_dir: str, use_pretrained: bool) -> DetectorOutput:
    model_dir_path = Path(model_dir)

    if use_pretrained:
        try:
            encoder = PretrainedEncoderDetector(
                encoder_name="wav2vec2-base",
                head_path=model_dir_path / "deepfake_pretrained_head.pt",
            )
            if encoder.available():
                return encoder.predict(samples, sr)
        except Exception:
            # Any failure (missing deps, missing weights, runtime error)
            # falls through to the CNN/classical path below -- the app
            # must never crash because an optional pretrained model is
            # unavailable.
            pass

    if TORCH_AVAILABLE:
        try:
            result = _run_cnn_detection(samples, sr, model_dir_path)
            if result is not None:
                return result
        except Exception:
            # A corrupt checkpoint or unexpected runtime error falls
            # through to the classical fallback rather than raising.
            pass

    baseline = _get_classical_baseline(model_dir_path)
    feature_vector = extract_baseline_feature_vector(samples, sr)
    return baseline.predict(feature_vector)

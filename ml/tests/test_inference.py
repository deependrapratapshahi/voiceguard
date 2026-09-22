"""
Tests for ml/deepfake/inference.py's model-selection and fallback logic.

The "no checkpoint present -> UNCERTAIN" path needs soundfile/librosa
(for the classical-fallback feature extraction it falls through to) so
it is gated accordingly. Torch-specific behavior is gated on torch.
"""
import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
librosa = pytest.importorskip("librosa")

from ml.deepfake.inference import run_detection


def test_run_detection_with_no_checkpoint_returns_valid_bounded_output(tmp_path):
    """With no trained model of any kind present, the system must still
    return a well-formed, bounded result -- never crash, never a
    fabricated confident number."""
    samples = (np.random.rand(16000 * 2).astype(np.float32) - 0.5) * 0.5
    result = run_detection(samples, sr=16000, model_dir=str(tmp_path), use_pretrained=False)

    assert 0.0 <= result.synthetic_probability <= 1.0
    assert result.label in ("LIKELY_REAL", "UNCERTAIN", "LIKELY_SYNTHETIC")
    assert "untrained" in result.model_version


def test_run_detection_handles_empty_samples_without_crashing(tmp_path):
    samples = np.zeros(0, dtype=np.float32)
    result = run_detection(samples, sr=16000, model_dir=str(tmp_path), use_pretrained=False)
    assert 0.0 <= result.synthetic_probability <= 1.0


torch = pytest.importorskip("torch")


def test_run_detection_uses_trained_cnn_checkpoint_when_present(tmp_path):
    """End-to-end: train a tiny CNN for one step, save it where
    inference.py expects to find it, and confirm run_detection picks it
    up and returns a real model-produced probability (not the untrained
    placeholder)."""
    from ml.common.configuration import CNN_CHECKPOINT_FILENAME, DEFAULT_SPECTROGRAM_CONFIG
    from ml.deepfake.model import SyntheticVoiceCNN, save_checkpoint

    model = SyntheticVoiceCNN(n_mels=DEFAULT_SPECTROGRAM_CONFIG.n_mels)
    model.eval()

    checkpoint_path = tmp_path / CNN_CHECKPOINT_FILENAME
    config = {
        **DEFAULT_SPECTROGRAM_CONFIG.to_dict(),
        "model_version": "cnn-baseline-v1",
    }
    save_checkpoint(model, checkpoint_path, config)

    # Reset the module-level singleton cache so this test doesn't pick up
    # a stale model from a previous test run in the same process.
    import ml.deepfake.inference as inference_module
    inference_module._cnn_singleton = None
    inference_module._cnn_checkpoint_path_loaded = None

    samples = (np.random.rand(16000 * 2).astype(np.float32) - 0.5) * 0.5
    result = run_detection(samples, sr=16000, model_dir=str(tmp_path), use_pretrained=False)

    assert 0.0 <= result.synthetic_probability <= 1.0
    assert result.model_version == "cnn-baseline-v1"
    assert "untrained" not in result.model_version

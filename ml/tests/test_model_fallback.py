"""
Tests that the deepfake model module degrades gracefully when PyTorch
is NOT installed -- these run in every environment, including this
sandbox where torch genuinely is absent, and directly exercise the
fallback behavior real deployments could hit if torch fails to install.
"""
from pathlib import Path

from ml.deepfake.model import (
    BaselineClassicalDetector,
    DetectorOutput,
    probability_to_label,
)


def test_probability_to_label_boundaries():
    assert probability_to_label(0.0) == "LIKELY_REAL"
    assert probability_to_label(0.34) == "LIKELY_REAL"
    assert probability_to_label(0.35) == "UNCERTAIN"
    assert probability_to_label(0.64) == "UNCERTAIN"
    assert probability_to_label(0.65) == "LIKELY_SYNTHETIC"
    assert probability_to_label(1.0) == "LIKELY_SYNTHETIC"


def test_classical_detector_untrained_returns_explicit_uncertain():
    """Never fabricate a confident score when no model has been trained."""
    detector = BaselineClassicalDetector(model_path=None)
    assert detector.is_trained() is False

    import numpy as np

    output = detector.predict(np.zeros(168, dtype=np.float32))
    assert output.label == "UNCERTAIN"
    assert output.synthetic_probability == 0.5
    assert output.model_version.endswith("-untrained")


def test_classical_detector_missing_file_path_is_untrained():
    detector = BaselineClassicalDetector(model_path=Path("/nonexistent/path.pkl"))
    assert detector.is_trained() is False


def test_synthetic_voice_cnn_raises_clear_import_error_without_torch():
    """If this test file is run in an environment where torch really is
    missing, instantiating the CNN must fail loudly and clearly rather
    than silently misbehaving."""
    from ml.deepfake.model import TORCH_AVAILABLE, SyntheticVoiceCNN

    if TORCH_AVAILABLE:
        import pytest

        pytest.skip("torch is installed in this environment; ImportError path not applicable")

    try:
        SyntheticVoiceCNN()
        assert False, "Expected ImportError when torch is unavailable"
    except ImportError as e:
        assert "PyTorch" in str(e)

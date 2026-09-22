import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.smoothing import SmoothingState
from ml.speaker.verification import calibrate_threshold, cosine_similarity, verify
import numpy as np


def test_ema_smoothing_converges_upward():
    state = SmoothingState(strategy="ema")
    values = [0.65, 0.71, 0.77, 0.83]
    outputs = [state.update(v) for v in values]
    assert outputs[-1] > outputs[0]
    assert outputs[-1] <= max(values)


def test_moving_average_smoothing_bounded():
    state = SmoothingState(strategy="moving_average")
    values = [0.1, 0.9, 0.1, 0.9, 0.1]
    outputs = [state.update(v) for v in values]
    assert all(0.0 <= o <= 1.0 for o in outputs)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) > 0.99


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    # orthogonal vectors -> raw cosine 0 -> mapped to 0.5
    assert abs(cosine_similarity(a, b) - 0.5) < 1e-6


def test_verify_respects_threshold():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    result = verify(a, b, threshold=0.9)
    assert result.identity_match is True

    c = np.array([0.0, 1.0, 0.0])
    result2 = verify(a, c, threshold=0.9)
    assert result2.identity_match is False


def test_calibrate_threshold_returns_reasonable_value():
    genuine = [0.8, 0.85, 0.9, 0.75]
    impostor = [0.2, 0.3, 0.25, 0.15]
    threshold = calibrate_threshold(genuine, impostor)
    assert 0.0 <= threshold <= 1.0
    assert min(genuine) >= threshold or threshold <= max(impostor) is False

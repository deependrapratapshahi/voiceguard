"""
Tests for ml/common/spectrogram_utils.py -- pure numpy, no librosa/torch
required, so these run in every environment including minimal CI images.
"""
import numpy as np

from ml.common.spectrogram_utils import pad_or_truncate_and_normalize


def test_pads_short_input_to_fixed_frames():
    short = np.random.rand(64, 50).astype(np.float32) * -40
    out = pad_or_truncate_and_normalize(short, fixed_frames=128, n_mels=64)
    assert out.shape == (64, 128)


def test_truncates_long_input_to_fixed_frames():
    long = np.random.rand(64, 200).astype(np.float32) * -40
    out = pad_or_truncate_and_normalize(long, fixed_frames=128, n_mels=64)
    assert out.shape == (64, 128)


def test_exact_length_input_unchanged_shape():
    exact = np.random.rand(64, 128).astype(np.float32) * -40
    out = pad_or_truncate_and_normalize(exact, fixed_frames=128, n_mels=64)
    assert out.shape == (64, 128)


def test_output_is_normalized():
    exact = np.random.rand(64, 128).astype(np.float32) * -40
    out = pad_or_truncate_and_normalize(exact, fixed_frames=128, n_mels=64)
    assert abs(np.mean(out)) < 1e-3
    assert abs(np.std(out) - 1.0) < 1e-3


def test_empty_input_does_not_crash():
    empty = np.zeros((64, 0), dtype=np.float32)
    out = pad_or_truncate_and_normalize(empty, fixed_frames=128, n_mels=64)
    assert out.shape == (64, 128)
    assert np.all(out == 0)


def test_output_dtype_is_float32():
    exact = np.random.rand(64, 128).astype(np.float64) * -40
    out = pad_or_truncate_and_normalize(exact, fixed_frames=128, n_mels=64)
    assert out.dtype == np.float32

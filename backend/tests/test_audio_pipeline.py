import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.audio_service import chunk_waveform, resample_and_normalize, simple_vad_mask
from app.utils.audio import (
    AudioTooShortError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    validate_size,
    validate_upload_filename,
)


def test_validate_upload_filename_accepts_wav():
    validate_upload_filename("sample.wav")  # should not raise


def test_validate_upload_filename_rejects_unknown_extension():
    with pytest.raises(UnsupportedAudioFormatError):
        validate_upload_filename("sample.exe")


def test_validate_size_rejects_empty():
    with pytest.raises(EmptyAudioError):
        validate_size(0, max_mb=25)


def test_validate_size_rejects_too_large():
    with pytest.raises(UnsupportedAudioFormatError):
        validate_size(30 * 1024 * 1024, max_mb=25)


def test_resample_and_normalize_peak_bounded():
    sr = 8000
    samples = (np.random.rand(sr) * 10 - 5).astype(np.float32)  # exceeds [-1, 1]
    out = resample_and_normalize(samples, sr)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_chunk_waveform_produces_expected_chunk_count():
    sr = 16000
    duration_seconds = 5
    samples = np.random.rand(sr * duration_seconds).astype(np.float32) * 0.1
    chunks = chunk_waveform(samples, sr)
    assert len(chunks) > 0
    for c in chunks:
        assert c.samples.shape[0] > 0
        assert c.end_time > c.start_time


def test_simple_vad_mask_silence_vs_signal():
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    loud = (np.random.rand(sr).astype(np.float32) - 0.5) * 2.0

    silence_mask = simple_vad_mask(silence)
    loud_mask = simple_vad_mask(loud)

    assert np.mean(silence_mask) < np.mean(loud_mask) + 1e-6

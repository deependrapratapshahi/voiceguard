"""
Tests for ml/deepfake/preprocessing.py's spectrogram-matrix building.

Requires soundfile + librosa to decode real (synthetic sine-wave) WAV
fixtures -- skipped cleanly via pytest.importorskip if unavailable.
No real voice data is used; fixtures are generated sine waves.
"""
import struct
import wave
from pathlib import Path

import pytest

sf = pytest.importorskip("soundfile")
librosa = pytest.importorskip("librosa")

from ml.common.configuration import SpectrogramConfig
from ml.deepfake.dataset import DatasetItem
from ml.deepfake.preprocessing import build_spectrogram_matrix, load_and_extract_spectrogram


def _write_sine_wav(path: Path, duration_seconds: float = 2.0, sr: int = 16000, freq: float = 220.0):
    n_samples = int(duration_seconds * sr)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            import math

            value = int(8000 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", value)
        wf.writeframes(bytes(frames))


def test_load_and_extract_spectrogram_returns_expected_shape(tmp_path):
    wav_path = tmp_path / "sample.wav"
    _write_sine_wav(wav_path)
    item = DatasetItem(path=wav_path, label=0, speaker_id="test_speaker")

    config = SpectrogramConfig(n_mels=64, fixed_frames=128)
    spec = load_and_extract_spectrogram(item, config)

    assert spec is not None
    assert spec.shape == (64, 128)


def test_load_and_extract_spectrogram_returns_none_for_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"this is not a valid wav file")
    item = DatasetItem(path=corrupt_path, label=0, speaker_id="test_speaker")

    result = load_and_extract_spectrogram(item)
    assert result is None


def test_build_spectrogram_matrix_stacks_correctly(tmp_path):
    items = []
    for i in range(3):
        wav_path = tmp_path / f"real_{i}.wav"
        _write_sine_wav(wav_path, freq=200 + i * 10)
        items.append(DatasetItem(path=wav_path, label=0, speaker_id=f"speaker{i}"))
    for i in range(2):
        wav_path = tmp_path / f"synth_{i}.wav"
        _write_sine_wav(wav_path, freq=400 + i * 10)
        items.append(DatasetItem(path=wav_path, label=1, speaker_id=f"synthspeaker{i}"))

    config = SpectrogramConfig(n_mels=64, fixed_frames=128)
    X, y, kept = build_spectrogram_matrix(items, config)

    assert X.shape == (5, 1, 64, 128)
    assert y.shape == (5,)
    assert len(kept) == 5
    assert set(y.tolist()) == {0.0, 1.0}


def test_build_spectrogram_matrix_skips_corrupt_files_without_crashing(tmp_path):
    good_path = tmp_path / "good.wav"
    _write_sine_wav(good_path)
    bad_path = tmp_path / "bad.wav"
    bad_path.write_bytes(b"garbage")

    items = [
        DatasetItem(path=good_path, label=0, speaker_id="a"),
        DatasetItem(path=bad_path, label=0, speaker_id="b"),
    ]
    X, y, kept = build_spectrogram_matrix(items)

    assert len(kept) == 1
    assert X.shape[0] == 1


def test_build_spectrogram_matrix_empty_input_returns_empty_arrays():
    X, y, kept = build_spectrogram_matrix([])
    assert X.shape[0] == 0
    assert y.shape[0] == 0
    assert kept == []

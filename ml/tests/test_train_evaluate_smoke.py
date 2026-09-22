"""
End-to-end smoke test for the training pipeline: generates a small set
of synthetic sine-wave WAV fixtures across multiple "speakers", runs
train.py's main() with a couple of quick epochs, confirms a checkpoint
is produced, then runs evaluate.py's main() against it and confirms it
reports real (not fabricated) metrics without crashing.

This is intentionally a smoke test, not an accuracy test -- with only a
handful of synthetic sine waves distinguishing "real" from "synthetic"
by frequency alone, the resulting accuracy numbers are not meaningful
and are not asserted on; only successful, crash-free execution and
correct output shapes/types are checked.
"""
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

sf = pytest.importorskip("soundfile")
librosa = pytest.importorskip("librosa")
torch = pytest.importorskip("torch")


def _write_sine_wav(path: Path, freq: float, duration_seconds: float = 1.5, sr: int = 16000):
    n_samples = int(duration_seconds * sr)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            value = int(8000 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", value)
        wf.writeframes(bytes(frames))


def test_train_then_evaluate_end_to_end(tmp_path, capsys):
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synth_dir.mkdir()

    # 8 "speakers" per class, 2 samples each -- enough for a
    # speaker-independent split to produce non-empty train/val/test.
    for s in range(8):
        for i in range(2):
            _write_sine_wav(real_dir / f"realspk{s}_{i}.wav", freq=150 + s * 5)
    for s in range(8):
        for i in range(2):
            _write_sine_wav(synth_dir / f"synthspk{s}_{i}.wav", freq=500 + s * 5)

    checkpoint_path = tmp_path / "artifacts" / "deepfake_cnn_baseline.pt"

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.deepfake import evaluate, train

    old_argv = sys.argv
    sys.argv = [
        "train.py",
        "--real-dir", str(real_dir),
        "--synthetic-dir", str(synth_dir),
        "--output", str(checkpoint_path),
        "--epochs", "2",
        "--batch-size", "4",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
    ]
    try:
        train_exit_code = train.main()
    finally:
        sys.argv = old_argv

    assert train_exit_code == 0
    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".json").exists()

    sys.argv = [
        "evaluate.py",
        "--real-dir", str(real_dir),
        "--synthetic-dir", str(synth_dir),
        "--model", str(checkpoint_path),
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
    ]
    try:
        eval_exit_code = evaluate.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert eval_exit_code == 0
    assert "Accuracy" in captured.out

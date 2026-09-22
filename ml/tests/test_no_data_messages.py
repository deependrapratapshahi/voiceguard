"""
Verifies that training/evaluation never invent results when no dataset
or no trained model is present -- they must exit cleanly with a clear
setup message instead. These tests only touch pure-Python logic
(argument parsing avoided; we call the underlying dataset functions
directly) so they run in every environment.
"""
from pathlib import Path

from ml.deepfake.dataset import dataset_statistics, load_directory_dataset


def test_empty_dataset_directories_produce_zero_stats(tmp_path):
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synth_dir.mkdir()

    items = load_directory_dataset(real_dir, synth_dir)
    stats = dataset_statistics(items)

    assert stats["total"] == 0
    assert stats["real"] == 0
    assert stats["synthetic"] == 0
    # This is exactly the condition train.py / evaluate.py check before
    # printing "No training data found..." instead of fabricating a result.


def test_evaluate_reports_missing_checkpoint_without_crashing(tmp_path, capsys):
    """evaluate.py must print a clear message and exit(1) if no
    checkpoint exists yet -- never crash, never invent metrics."""
    import sys

    import pytest

    pytest.importorskip("soundfile")  # ml/deepfake/preprocessing.py requires it at import time
    pytest.importorskip("librosa")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.deepfake import evaluate

    nonexistent_model = tmp_path / "does_not_exist.pt"
    args = [
        "evaluate.py",
        "--model", str(nonexistent_model),
        "--real-dir", str(tmp_path / "real"),
        "--synthetic-dir", str(tmp_path / "synthetic"),
    ]
    old_argv = sys.argv
    sys.argv = args
    try:
        exit_code = evaluate.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert exit_code == 1
    # Either "PyTorch is not installed" (this environment) or
    # "No trained model found" (environment with torch but no checkpoint)
    assert "PyTorch is not installed" in captured.out or "No trained model found" in captured.out

"""
Tests for ml/deepfake/dataset.py -- pure Python (no audio libraries
needed), so these exercise the actual dataset-loading and,
critically, the speaker-leakage-prevention logic in every environment.
"""
from pathlib import Path

from ml.deepfake.dataset import (
    DatasetItem,
    dataset_statistics,
    load_directory_dataset,
    speaker_independent_split,
)


def _make_items(n_real_speakers=6, n_synth_speakers=6, samples_per_speaker=3):
    items = []
    for s in range(n_real_speakers):
        for i in range(samples_per_speaker):
            items.append(
                DatasetItem(path=Path(f"real_speaker{s}_{i}.wav"), label=0, speaker_id=f"real_speaker{s}")
            )
    for s in range(n_synth_speakers):
        for i in range(samples_per_speaker):
            items.append(
                DatasetItem(path=Path(f"synth_speaker{s}_{i}.wav"), label=1, speaker_id=f"synth_speaker{s}")
            )
    return items


def test_load_directory_dataset_reads_real_and_synthetic(tmp_path):
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synth_dir.mkdir()

    (real_dir / "alice_001.wav").write_bytes(b"fake wav bytes")
    (real_dir / "bob_001.wav").write_bytes(b"fake wav bytes")
    (synth_dir / "clone_alice_001.wav").write_bytes(b"fake wav bytes")

    items = load_directory_dataset(real_dir, synth_dir)
    assert len(items) == 3
    labels = sorted(i.label for i in items)
    assert labels == [0, 0, 1]


def test_load_directory_dataset_ignores_non_audio_files(tmp_path):
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synth_dir.mkdir()

    (real_dir / "alice_001.wav").write_bytes(b"fake wav bytes")
    (real_dir / "readme.txt").write_text("not audio")

    items = load_directory_dataset(real_dir, synth_dir)
    assert len(items) == 1


def test_dataset_statistics_counts_correctly():
    items = _make_items(n_real_speakers=4, n_synth_speakers=2, samples_per_speaker=5)
    stats = dataset_statistics(items)
    assert stats["total"] == 30
    assert stats["real"] == 20
    assert stats["synthetic"] == 10
    assert stats["unique_speakers"] == 6


def test_speaker_independent_split_has_no_leakage():
    """The critical requirement: no speaker_id should ever appear in
    more than one of train/val/test."""
    items = _make_items(n_real_speakers=10, n_synth_speakers=10, samples_per_speaker=4)
    train, val, test = speaker_independent_split(items, val_ratio=0.2, test_ratio=0.2, seed=1)

    train_speakers = {i.speaker_id for i in train}
    val_speakers = {i.speaker_id for i in val}
    test_speakers = {i.speaker_id for i in test}

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    # every sample from a given speaker must land in exactly one split
    all_items = train + val + test
    assert len(all_items) == len(items)


def test_speaker_independent_split_is_deterministic_given_seed():
    items = _make_items(n_real_speakers=8, n_synth_speakers=8, samples_per_speaker=2)
    split_a = speaker_independent_split(items, seed=7)
    split_b = speaker_independent_split(items, seed=7)

    speakers_a = [sorted({i.speaker_id for i in group}) for group in split_a]
    speakers_b = [sorted({i.speaker_id for i in group}) for group in split_b]
    assert speakers_a == speakers_b


def test_speaker_independent_split_produces_nonempty_train_with_enough_speakers():
    items = _make_items(n_real_speakers=10, n_synth_speakers=10, samples_per_speaker=3)
    train, val, test = speaker_independent_split(items, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert len(train) > 0
    # with 20 total speakers and 15%/15% ratios, val/test should be non-trivial
    assert len(val) > 0
    assert len(test) > 0


def test_speaker_independent_split_handles_tiny_dataset_without_crashing():
    items = _make_items(n_real_speakers=1, n_synth_speakers=1, samples_per_speaker=2)
    train, val, test = speaker_independent_split(items)
    # Should not raise, and every item should be accounted for exactly once.
    assert len(train) + len(val) + len(test) == len(items)

"""
Directory/metadata-based dataset loader for the deepfake detector, with
speaker-independent train/val/test splitting to avoid data leakage.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


@dataclass
class DatasetItem:
    path: Path
    label: int  # 0 = real, 1 = synthetic
    speaker_id: str


def _infer_speaker_id(path: Path) -> str:
    stem = path.stem
    for sep in ("_", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


def load_directory_dataset(real_dir: Path, synthetic_dir: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for f in sorted(real_dir.rglob("*")):
        if f.suffix.lower() in AUDIO_EXTS:
            items.append(DatasetItem(path=f, label=0, speaker_id=_infer_speaker_id(f)))
    for f in sorted(synthetic_dir.rglob("*")):
        if f.suffix.lower() in AUDIO_EXTS:
            items.append(DatasetItem(path=f, label=1, speaker_id=_infer_speaker_id(f)))
    return items


def load_metadata_dataset(metadata_path: Path, audio_root: Path) -> list[DatasetItem]:
    """
    Supports a metadata CSV/JSON with columns/fields:
    file_path, label (real|synthetic or 0|1), speaker_id
    """
    items: list[DatasetItem] = []

    if metadata_path.suffix.lower() == ".csv":
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                items.append(_row_to_item(row, audio_root))
    elif metadata_path.suffix.lower() == ".json":
        with open(metadata_path) as f:
            for row in json.load(f):
                items.append(_row_to_item(row, audio_root))
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

    return items


def _row_to_item(row: dict, audio_root: Path) -> DatasetItem:
    label_raw = str(row["label"]).lower()
    label = 1 if label_raw in ("1", "synthetic", "fake", "spoof") else 0
    return DatasetItem(
        path=audio_root / row["file_path"],
        label=label,
        speaker_id=row.get("speaker_id") or _infer_speaker_id(Path(row["file_path"])),
    )


def speaker_independent_split(
    items: list[DatasetItem], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> tuple[list[DatasetItem], list[DatasetItem], list[DatasetItem]]:
    """
    Splits by speaker_id (not by individual sample) so no speaker
    appears in more than one split -- this avoids the data leakage the
    project spec explicitly calls out.
    """
    speakers = sorted({item.speaker_id for item in items})
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(n * test_ratio)) if n > 2 else 0
    n_val = max(1, int(n * val_ratio)) if n > 2 else 0

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test:n_test + n_val])
    train_speakers = set(speakers[n_test + n_val:])

    train = [i for i in items if i.speaker_id in train_speakers]
    val = [i for i in items if i.speaker_id in val_speakers]
    test = [i for i in items if i.speaker_id in test_speakers]
    return train, val, test


def dataset_statistics(items: list[DatasetItem]) -> dict:
    n_real = sum(1 for i in items if i.label == 0)
    n_synth = sum(1 for i in items if i.label == 1)
    n_speakers = len({i.speaker_id for i in items})
    return {
        "total": len(items),
        "real": n_real,
        "synthetic": n_synth,
        "unique_speakers": n_speakers,
    }

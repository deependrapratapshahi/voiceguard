"""
Validates the expected directory-based dataset layout for deepfake
detector training:

    datasets/real/<speaker_or_file>.wav
    datasets/synthetic/<speaker_or_file>.wav
    datasets/metadata/*.csv | *.json   (optional speaker/split metadata)

Reports class balance, basic file-integrity issues, and warns about
possible train/test speaker leakage if speaker IDs can be inferred from
filenames (e.g. "speakerA_001.wav").

This script does NOT download any datasets. Place ASVspoof or other
legally obtained anti-spoofing datasets into datasets/real and
datasets/synthetic yourself, respecting each dataset's license.
"""
from __future__ import annotations

import sys
import wave
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = REPO_ROOT / "datasets" / "real"
SYNTH_DIR = REPO_ROOT / "datasets" / "synthetic"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def infer_speaker_id(path: Path) -> str:
    # Heuristic: text before the first underscore or hyphen.
    stem = path.stem
    for sep in ("_", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


def scan_dir(directory: Path) -> tuple[list[Path], list[str]]:
    issues = []
    if not directory.exists():
        issues.append(f"Missing directory: {directory}")
        return [], issues

    files = [p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    for f in files:
        try:
            if f.stat().st_size == 0:
                issues.append(f"Empty file: {f}")
                continue
            if f.suffix.lower() == ".wav":
                with wave.open(str(f), "rb") as wf:
                    if wf.getnframes() == 0:
                        issues.append(f"Zero-length WAV: {f}")
        except Exception as exc:
            issues.append(f"Corrupt/unreadable file: {f} ({exc})")

    return files, issues


def main() -> int:
    real_files, real_issues = scan_dir(REAL_DIR)
    synth_files, synth_issues = scan_dir(SYNTH_DIR)

    print("=== VOICEGUARD Dataset Validation ===")
    print(f"Real samples:      {len(real_files)}")
    print(f"Synthetic samples: {len(synth_files)}")

    total = len(real_files) + len(synth_files)
    if total == 0:
        print("\nNo audio files found yet.")
        print(f"Place real speech in:      {REAL_DIR}")
        print(f"Place synthetic speech in: {SYNTH_DIR}")
        return 0

    real_pct = 100 * len(real_files) / total
    synth_pct = 100 * len(synth_files) / total
    print(f"Class balance: {real_pct:.1f}% real / {synth_pct:.1f}% synthetic")
    if abs(real_pct - synth_pct) > 20:
        print("WARNING: significant class imbalance detected.")

    speaker_map: dict[str, set[str]] = defaultdict(set)
    for f in real_files:
        speaker_map[infer_speaker_id(f)].add("real")
    for f in synth_files:
        speaker_map[infer_speaker_id(f)].add("synthetic")

    overlapping = [s for s, labels in speaker_map.items() if len(labels) > 1]
    if overlapping:
        print(
            f"\nNOTE: {len(overlapping)} inferred speaker ID(s) appear in both classes "
            "(this is expected for many anti-spoofing datasets where the same "
            "speaker has both bona fide and spoofed samples -- just make sure "
            "your train/val/test SPLITS keep each speaker's samples together, "
            "see ml/deepfake/dataset.py)."
        )

    all_issues = real_issues + synth_issues
    if all_issues:
        print(f"\n{len(all_issues)} issue(s) found:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1

    print("\nNo integrity issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

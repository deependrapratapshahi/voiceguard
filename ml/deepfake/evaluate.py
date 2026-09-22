"""
Evaluates a trained CNN baseline checkpoint on the held-out
(speaker-independent) test split. Reports accuracy, precision, recall,
F1, and ROC-AUC -- all computed directly from scikit-learn, never
invented. If no trained checkpoint or no test data is available, this
script says so explicitly instead of printing placeholder numbers.

Usage:
    python -m ml.deepfake.evaluate \
        --real-dir datasets/real --synthetic-dir datasets/synthetic \
        --model ml_artifacts/deepfake_cnn_baseline.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.common.configuration import SpectrogramConfig
from ml.common.metrics import compute_classification_metrics, format_metrics_report
from ml.deepfake.dataset import load_directory_dataset, speaker_independent_split

try:
    from ml.deepfake.preprocessing import build_spectrogram_matrix

    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

try:
    import torch

    from ml.deepfake.model import load_checkpoint

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=Path("datasets/real"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--model", type=Path, default=Path("ml_artifacts/deepfake_cnn_baseline.pt"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not TORCH_AVAILABLE:
        print("PyTorch is not installed in this environment; cannot load or evaluate the CNN checkpoint.")
        return 1

    if not args.model.exists():
        print(f"No trained model found at {args.model}. Run `python -m ml.deepfake.train` first.")
        return 1

    if not PREPROCESSING_AVAILABLE:
        print(
            "librosa and/or soundfile are not installed in this environment. "
            "Install them via backend/requirements.txt to extract audio features."
        )
        return 1

    model, config_dict = load_checkpoint(args.model)
    config = SpectrogramConfig.from_dict(config_dict)
    print(f"Loaded checkpoint: model_version={config_dict.get('model_version')}, "
          f"trained_at={config_dict.get('trained_at')}")

    items = load_directory_dataset(args.real_dir, args.synthetic_dir)
    if not items:
        print("No dataset found to evaluate against. Populate datasets/real and datasets/synthetic first.")
        return 1

    _train, _val, test_items = speaker_independent_split(
        items, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    if not test_items:
        print("Test split is empty (dataset too small for a held-out speaker split). "
              "Add more speakers or lower --test-ratio.")
        return 1

    X_test, y_test, kept = build_spectrogram_matrix(test_items, config)
    if len(kept) == 0:
        print("No usable test files after feature extraction.")
        return 1

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = compute_classification_metrics(y_test.astype(int), preds, probs)
    print(f"\nTest set: {len(kept)} samples ({int(np.sum(y_test == 0))} real / {int(np.sum(y_test == 1))} synthetic)")
    print(format_metrics_report(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

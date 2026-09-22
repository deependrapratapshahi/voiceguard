"""
Trains the CNN baseline synthetic-voice detector (SyntheticVoiceCNN)
over fixed-size log-mel spectrograms, using a speaker-independent
train/validation/test split to avoid data leakage.

Usage:
    python -m ml.deepfake.train \
        --real-dir datasets/real --synthetic-dir datasets/synthetic \
        --output ml_artifacts/deepfake_cnn_baseline.pt \
        --epochs 15 --batch-size 8

This does NOT download or fabricate any dataset. Populate
datasets/real/ and datasets/synthetic/ yourself with legally obtained
audio (see docs/ml_pipeline.md) before running this script -- if no
data is found, a clear setup message is printed and nothing is trained
or saved.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml.common.configuration import CNN_MODEL_VERSION, DEFAULT_SPECTROGRAM_CONFIG
from ml.common.metrics import compute_classification_metrics, format_metrics_report
from ml.deepfake.dataset import (
    dataset_statistics,
    load_directory_dataset,
    speaker_independent_split,
)

try:
    from ml.deepfake.preprocessing import build_spectrogram_matrix

    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from ml.deepfake.model import SyntheticVoiceCNN, save_checkpoint

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=Path("datasets/real"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("ml_artifacts/deepfake_cnn_baseline.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _no_data_message(real_dir: Path, synthetic_dir: Path) -> None:
    print(
        "No training data found.\n"
        f"  Place real (bona fide) speech in:      {real_dir}\n"
        f"  Place synthetic/cloned speech in:       {synthetic_dir}\n"
        "Then re-run this script. VOICEGUARD does not download or fabricate "
        "any dataset automatically -- see docs/ml_pipeline.md for guidance "
        "on obtaining a legally usable anti-spoofing dataset (e.g. ASVspoof). "
        "Until a model is trained, the detection API will report UNCERTAIN "
        "results labeled as an untrained baseline rather than inventing a "
        "confident score."
    )


def main() -> int:
    args = parse_args()

    if not TORCH_AVAILABLE:
        print(
            "PyTorch is not installed in this environment. Install it via "
            "backend/requirements.txt (`pip install torch torchaudio`) to "
            "train the CNN baseline. The classical fallback detector will "
            "continue to serve UNCERTAIN results until then."
        )
        return 1

    items = load_directory_dataset(args.real_dir, args.synthetic_dir)
    stats = dataset_statistics(items)
    print(f"Dataset stats: {stats}")

    if stats["total"] == 0:
        _no_data_message(args.real_dir, args.synthetic_dir)
        return 0

    if not PREPROCESSING_AVAILABLE:
        print(
            "librosa and/or soundfile are not installed in this environment. "
            "Install them via backend/requirements.txt (`pip install librosa soundfile`) "
            "to extract audio features and train the model."
        )
        return 1

    train_items, val_items, test_items = speaker_independent_split(
        items, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    print(
        f"Speaker-independent split: {len(train_items)} train / "
        f"{len(val_items)} val / {len(test_items)} test"
    )

    config = DEFAULT_SPECTROGRAM_CONFIG
    X_train, y_train, kept_train = build_spectrogram_matrix(train_items, config)
    X_val, y_val, kept_val = build_spectrogram_matrix(val_items, config)

    if len(kept_train) == 0:
        print("No usable training files after feature extraction (all failed to load).")
        return 1

    if len(set(y_train.tolist())) < 2:
        print(
            "Training set only contains one class after the speaker-independent "
            "split. Add more speakers/samples to both datasets/real and "
            "datasets/synthetic, or adjust --val-ratio/--test-ratio."
        )
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = SyntheticVoiceCNN(n_mels=config.n_mels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    has_val = len(kept_val) > 0 and len(set(y_val.tolist())) >= 1
    val_tensor_x = torch.from_numpy(X_val).to(device) if has_val else None
    val_tensor_y = y_val if has_val else None

    best_val_metric = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        log_line = f"Epoch {epoch}/{args.epochs} - train_loss={avg_loss:.4f}"

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_tensor_x)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_metrics = compute_classification_metrics(val_tensor_y.astype(int), val_preds, val_probs)
            tracked = val_metrics["f1"] if val_metrics["f1"] is not None else val_metrics["accuracy"]
            log_line += f" - val_acc={val_metrics['accuracy']:.3f} - val_f1={val_metrics['f1']:.3f}"

            if tracked is not None and tracked >= best_val_metric:
                best_val_metric = tracked
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(log_line)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint by validation F1 ({best_val_metric:.3f}).")
    else:
        print("No validation set available; saving the final-epoch weights.")

    checkpoint_config = {
        **config.to_dict(),
        "model_version": CNN_MODEL_VERSION,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train": len(kept_train),
        "n_val": len(kept_val),
        "n_test": len(test_items),
    }
    model_cpu = model.to("cpu")
    save_checkpoint(model_cpu, args.output, checkpoint_config)
    metadata_path = args.output.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(checkpoint_config, f, indent=2)

    print(f"\nSaved trained CNN checkpoint to {args.output}")
    print(f"Saved run metadata to {metadata_path}")
    print(
        "Run `python -m ml.deepfake.evaluate` against the held-out test "
        "split to get honest generalization numbers before trusting this model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

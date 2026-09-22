"""
Synthetic-voice ("deepfake") detector model definitions.

Two implementations are provided behind a common `DetectorOutput` shape:

1. SyntheticVoiceCNN -- the primary baseline: a small PyTorch CNN over
   fixed-size log-mel spectrograms (see ml/common/audio.py
   extract_log_mel_fixed). This is intentionally small and simple (a
   genuine "baseline", not a state-of-the-art architecture) but is a
   real, trainable neural network -- not a placeholder.

2. BaselineClassicalDetector -- a scikit-learn logistic-regression
   classifier over summarized log-mel + MFCC features. This exists as a
   defensive fallback for environments where PyTorch itself is
   unavailable, per the project's "the app must keep working even if
   advanced models are unavailable" requirement. It is not the primary
   detector once a CNN checkpoint has been trained.

Both implementations output calibrated-looking probabilities, never a
hard "real"/"fake" certainty claim. Neither path fabricates a result:
if no trained weights exist, callers get an explicit UNCERTAIN /
"-untrained" result rather than an invented confident number.

A third, optional path (PretrainedEncoderDetector) remains available
for wiring in a pretrained self-supervised speech encoder later.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.common.configuration import CLASSICAL_FALLBACK_VERSION

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None
    nn = None
    TORCH_AVAILABLE = False


@dataclass
class DetectorOutput:
    synthetic_probability: float
    label: str
    model_version: str


def probability_to_label(p: float) -> str:
    if p < 0.35:
        return "LIKELY_REAL"
    if p < 0.65:
        return "UNCERTAIN"
    return "LIKELY_SYNTHETIC"


# ---------------------------------------------------------------------
# Primary baseline: small CNN over fixed-size log-mel spectrograms
# ---------------------------------------------------------------------

if TORCH_AVAILABLE:

    class SyntheticVoiceCNN(nn.Module):
        """
        Small CNN baseline over fixed-size log-mel spectrograms.

        Input:  (batch, 1, n_mels, fixed_frames)
        Output: raw logits, shape (batch,) -- apply sigmoid to get a
                synthetic-speech probability in [0, 1].

        Architecture: three Conv2d/BatchNorm/ReLU/MaxPool blocks,
        global average pooling, then a small MLP head. Deliberately
        small (~40k parameters) so it trains quickly on a normal
        development machine without a GPU -- this is a baseline meant
        to be replaced by a stronger architecture (e.g. a pretrained
        encoder) for production-grade accuracy, not a final model.
        """

        def __init__(self, n_mels: int = 64):
            super().__init__()
            self.n_mels = n_mels
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            x = self.features(x)
            logits = self.classifier(x)
            return logits.squeeze(-1)  # (batch,) raw logits

    def save_checkpoint(model: "SyntheticVoiceCNN", path: Path, config: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config}, path)

    def load_checkpoint(path: Path) -> tuple["SyntheticVoiceCNN", dict]:
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        model = SyntheticVoiceCNN(n_mels=config.get("n_mels", 64))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, config

else:  # pragma: no cover - exercised only when torch is absent

    class SyntheticVoiceCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for SyntheticVoiceCNN but is not installed. "
                "Install it via backend/requirements.txt. The classical fallback "
                "detector (BaselineClassicalDetector) will be used until then."
            )

    def save_checkpoint(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required to save a CNN checkpoint.")

    def load_checkpoint(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required to load a CNN checkpoint.")


# ---------------------------------------------------------------------
# Classical fallback: logistic regression over summarized features
# ---------------------------------------------------------------------

class BaselineClassicalDetector:
    """
    Logistic-regression baseline over summarized spectral features.
    Used only as a fallback when PyTorch is unavailable at runtime --
    see ml/deepfake/inference.py for the selection logic. Interpretable
    and dependency-light (scikit-learn only).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model: Optional[LogisticRegression] = None
        self.model_version = CLASSICAL_FALLBACK_VERSION
        if model_path and model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, feature_vector: np.ndarray) -> DetectorOutput:
        if self.model is None:
            return DetectorOutput(
                synthetic_probability=0.5,
                label="UNCERTAIN",
                model_version=f"{self.model_version}-untrained",
            )
        proba = float(self.model.predict_proba(feature_vector.reshape(1, -1))[0][1])
        return DetectorOutput(
            synthetic_probability=proba,
            label=probability_to_label(proba),
            model_version=self.model_version,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


# ---------------------------------------------------------------------
# Optional: pretrained self-supervised encoder interface (unchanged
# scope from the original scaffold -- not wired to real weights here)
# ---------------------------------------------------------------------

class PretrainedEncoderDetector:
    """
    Optional interface for a pretrained self-supervised speech encoder
    (wav2vec2/HuBERT/WavLM) plus a small trainable classification head.
    Only activated when weights are present locally; see
    ml/deepfake/inference.py for the fallback logic and
    scripts/download_models.py for how a developer would obtain weights
    (not fetched automatically, out of respect for model licensing).
    """

    def __init__(self, encoder_name: str, head_path: Optional[Path] = None):
        self.encoder_name = encoder_name
        self.head_path = head_path
        self.model_version = f"pretrained-{encoder_name}-v1"

    def available(self) -> bool:
        return self.head_path is not None and self.head_path.exists()

    def predict(self, samples: np.ndarray, sr: int) -> DetectorOutput:
        raise NotImplementedError(
            "PretrainedEncoderDetector.predict must be wired to a real "
            "downloaded encoder + trained head before use. See "
            "ml/deepfake/inference.py for the fallback path used until then."
        )

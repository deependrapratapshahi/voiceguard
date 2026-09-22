"""
Shared configuration for the deepfake detection ML pipeline.

Centralized here so training, evaluation, and inference always agree on
feature dimensions and model version strings -- a mismatch between
train-time and inference-time feature shapes is one of the most common,
hardest-to-debug bugs in ML pipelines, so it is defined exactly once.
"""
from dataclasses import asdict, dataclass

# Model version identifiers. Update these when the architecture changes
# in a way that makes old checkpoints incompatible.
CNN_MODEL_VERSION = "cnn-baseline-v1"
CLASSICAL_FALLBACK_VERSION = "classical-fallback-v1"

CNN_CHECKPOINT_FILENAME = "deepfake_cnn_baseline.pt"
CLASSICAL_CHECKPOINT_FILENAME = "deepfake_baseline.pkl"


@dataclass
class SpectrogramConfig:
    """Feature extraction parameters for the log-mel spectrogram fed to
    the CNN baseline. `fixed_frames=128` at `hop_length=256`/`sr=16000`
    corresponds to roughly 2.05 seconds of audio, matching the default
    streaming chunk duration (see backend/app/config.py CHUNK_DURATION_SECONDS)."""

    n_mels: int = 64
    fixed_frames: int = 128
    hop_length: int = 256
    n_fft: int = 1024
    sample_rate: int = 16000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SpectrogramConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


DEFAULT_SPECTROGRAM_CONFIG = SpectrogramConfig()

from pathlib import Path

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import ModelInfo

router = APIRouter(prefix="/api/v1/models", tags=["models"])
settings = get_settings()

CNN_CHECKPOINT_FILENAME = "deepfake_cnn_baseline.pt"


def _deepfake_model_info() -> ModelInfo:
    """
    Reports the deepfake detector's REAL current state rather than a
    static config value: whether a trained CNN checkpoint actually
    exists on disk. This avoids ever claiming a trained model is active
    when the system is really still serving untrained/uncertain results.
    """
    checkpoint_path = Path(settings.MODEL_PATH) / CNN_CHECKPOINT_FILENAME
    is_trained = checkpoint_path.exists()

    if settings.USE_PRETRAINED_ENCODERS:
        name = "pretrained-encoder"
    elif is_trained:
        name = "cnn-baseline (trained)"
    else:
        name = "cnn-baseline (untrained - serving UNCERTAIN until trained)"

    version = settings.DEEPFAKE_MODEL_VERSION if is_trained else f"{settings.DEEPFAKE_MODEL_VERSION}-untrained"

    return ModelInfo(
        component="deepfake",
        name=name,
        version=version,
        is_active=True,
        is_baseline=not settings.USE_PRETRAINED_ENCODERS,
    )


@router.get("", response_model=list[ModelInfo])
async def get_active_models():
    return [
        _deepfake_model_info(),
        ModelInfo(
            component="speaker",
            name="baseline-embedding",
            version=settings.SPEAKER_MODEL_VERSION,
            is_active=True,
            is_baseline=True,
        ),
        ModelInfo(
            component="prosody",
            name="rule-based-anomaly",
            version=settings.PROSODY_MODEL_VERSION,
            is_active=True,
            is_baseline=True,
        ),
    ]

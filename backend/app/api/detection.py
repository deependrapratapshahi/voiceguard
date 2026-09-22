from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.services import detection_service, prosody_service
from app.services.audio_service import preprocess_full_audio
from app.utils.audio import (
    AudioTooShortError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    validate_content_type,
    validate_size,
    validate_upload_filename,
)
from app.utils.logging import audit_log, get_logger

router = APIRouter(prefix="/api/v1/detection", tags=["detection"])
settings = get_settings()
logger = get_logger(__name__)


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        validate_upload_filename(file.filename or "")
        validate_content_type(file.content_type)
        raw_bytes = await file.read()
        validate_size(len(raw_bytes), settings.MAX_UPLOAD_MB)

        chunks = preprocess_full_audio(raw_bytes)
        if not chunks:
            raise AudioTooShortError("Audio too short after preprocessing to analyze.")

        detection_results = [detection_service.analyze_chunk(c) for c in chunks]
        prosody_results = [prosody_service.analyze_chunk(c) for c in chunks]

        avg_synthetic = sum(r.synthetic_probability for r in detection_results) / len(detection_results)
        avg_prosody = sum(r.prosody_anomaly for r in prosody_results) / len(prosody_results)

        audit_log("upload_analysis_complete", num_chunks=len(chunks))

        return {
            "num_chunks_analyzed": len(chunks),
            "average_synthetic_probability": round(avg_synthetic, 4),
            "average_prosody_anomaly": round(avg_prosody, 4),
            "per_chunk_detection": [r.model_dump() for r in detection_results],
            "per_chunk_prosody": [r.model_dump() for r in prosody_results],
        }

    except (UnsupportedAudioFormatError, EmptyAudioError, AudioTooShortError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error during audio analysis")
        raise HTTPException(status_code=500, detail="An internal error occurred while analyzing the audio.")

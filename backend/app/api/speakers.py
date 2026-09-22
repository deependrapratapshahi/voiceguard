from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.entities import Speaker
from app.models.schemas import SpeakerResponse, SpeakerVerificationResult
from app.services import speaker_service
from app.services.audio_service import preprocess_full_audio
from app.utils.audio import (
    AudioTooShortError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    validate_content_type,
    validate_size,
    validate_upload_filename,
)
from app.config import get_settings

router = APIRouter(prefix="/api/v1/speakers", tags=["speakers"])
settings = get_settings()


@router.post("", response_model=SpeakerResponse)
async def register_speaker(
    speaker_id: str,
    display_name: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        validate_upload_filename(file.filename or "")
        validate_content_type(file.content_type)
        raw_bytes = await file.read()
        validate_size(len(raw_bytes), settings.MAX_UPLOAD_MB)

        chunks = preprocess_full_audio(raw_bytes)
        if not chunks:
            raise AudioTooShortError("Reference audio too short to build an embedding.")

        embedding, model_version = speaker_service.register_reference(chunks[0])

        speaker = Speaker(
            speaker_id=speaker_id,
            display_name=display_name,
            embedding_model_version=model_version,
            reference_embedding=embedding,
        )
        db.add(speaker)
        await db.commit()
        await db.refresh(speaker)
        return speaker

    except (UnsupportedAudioFormatError, EmptyAudioError, AudioTooShortError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{speaker_id}/verify", response_model=SpeakerVerificationResult)
async def verify_speaker(
    speaker_id: str, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Speaker).where(Speaker.speaker_id == speaker_id))
    speaker = result.scalar_one_or_none()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker profile not found")

    try:
        validate_upload_filename(file.filename or "")
        validate_content_type(file.content_type)
        raw_bytes = await file.read()
        validate_size(len(raw_bytes), settings.MAX_UPLOAD_MB)

        chunks = preprocess_full_audio(raw_bytes)
        if not chunks:
            raise AudioTooShortError("Audio too short to verify against reference.")

        return speaker_service.verify_chunk(speaker.reference_embedding, chunks[0])

    except (UnsupportedAudioFormatError, EmptyAudioError, AudioTooShortError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

"""
Low-level audio helper utilities shared across services.

Real signal-processing logic (resampling, VAD, feature extraction) lives
in app/services/audio_service.py and ml/common/audio.py. This module
holds small, dependency-light helpers (validation, format checks).
"""
from __future__ import annotations

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ALLOWED_MIME_PREFIXES = ("audio/",)


class UnsupportedAudioFormatError(Exception):
    pass


class AudioTooShortError(Exception):
    pass


class EmptyAudioError(Exception):
    pass


def validate_upload_filename(filename: str) -> None:
    lowered = filename.lower()
    if not any(lowered.endswith(ext) for ext in ALLOWED_AUDIO_EXTENSIONS):
        raise UnsupportedAudioFormatError(
            f"Unsupported file extension. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}"
        )


def validate_content_type(content_type: str | None) -> None:
    if not content_type or not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise UnsupportedAudioFormatError(f"Unsupported content type: {content_type}")


def validate_size(num_bytes: int, max_mb: int) -> None:
    max_bytes = max_mb * 1024 * 1024
    if num_bytes <= 0:
        raise EmptyAudioError("Uploaded audio file is empty.")
    if num_bytes > max_bytes:
        raise UnsupportedAudioFormatError(f"File exceeds maximum size of {max_mb}MB.")

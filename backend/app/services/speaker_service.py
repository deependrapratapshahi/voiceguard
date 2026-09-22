from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(ML_ROOT.parent) not in sys.path:
    sys.path.append(str(ML_ROOT.parent))

from ml.speaker.embeddings import EMBEDDING_MODEL_VERSION  # noqa: E402
from ml.speaker.inference import embed, verify_against_reference  # noqa: E402
from ml.speaker.verification import DEFAULT_SIMILARITY_THRESHOLD  # noqa: E402

from app.models.schemas import SpeakerVerificationResult
from app.services.audio_service import AudioChunk


def register_reference(chunk: AudioChunk) -> tuple[list[float], str]:
    embedding = embed(chunk.samples, chunk.sample_rate)
    return embedding.tolist(), EMBEDDING_MODEL_VERSION


def verify_chunk(
    reference_embedding: list[float], chunk: AudioChunk, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> SpeakerVerificationResult:
    result = verify_against_reference(reference_embedding, chunk.samples, chunk.sample_rate, threshold)
    return SpeakerVerificationResult(
        speaker_similarity=result.speaker_similarity,
        identity_match=result.identity_match,
        confidence=result.confidence,
    )

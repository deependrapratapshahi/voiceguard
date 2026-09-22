"""
Near-real-time WebSocket audio detection pipeline: /ws/calls/{call_id}/audio

Pipeline per audio chunk, in order:
    audio decoding -> preprocessing -> VAD -> synthetic voice inference
    -> speaker verification -> prosody analysis -> risk engine
    -> temporal smoothing (applied to the final risk score, for a
       stable, non-jittery number the dashboard can trust)

The client sends raw audio bytes (self-contained audio chunks, e.g.
small WAV files) as binary WebSocket frames. Context factors (e.g.
"transaction requested") can be updated mid-call via a JSON text frame.
A JSON result is sent back after every successfully processed,
voiced chunk. Invalid/corrupt audio never terminates the connection --
a chunk-level error is reported and the stream continues.

IMPORTANT: All scores here are probabilistic risk indicators, not
proof of fraud -- see the `disclaimer` field on every result.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_db
from app.models.entities import Call, RiskEvent, Speaker
from app.models.schemas import ContextFactors, RiskResult, RiskWeights
from app.services import detection_service, prosody_service, speaker_service
from app.services.alert_service import maybe_generate_alert
from app.services.audio_service import AudioChunk, StreamingPreprocessor
from app.services.risk_engine import (
    compute_context_risk_summary,
    compute_risk,
    determine_recommended_action,
    score_to_level,
    weights_from_settings,
)
from app.services.smoothing import smoothing_registry
from app.utils.logging import audit_log, get_logger

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)

RISK_DISCLAIMER = (
    "This is a probabilistic risk indicator based on automated signal analysis, "
    "not proof of fraud or identity theft. Use recommended_action as guidance for "
    "additional verification, not as a final determination."
)


async def _load_reference_embedding(db: AsyncSession, call_id: str) -> Optional[list]:
    """
    If this call is linked to a registered speaker profile, load that
    speaker's reference embedding so speaker verification can run on
    each chunk. Returns None (verification simply skipped) if the call
    or speaker isn't found -- a missing reference is not an error.
    """
    try:
        result = await db.execute(select(Call).where(Call.call_id == call_id))
        call = result.scalar_one_or_none()
        if call is None or not call.speaker_id:
            return None

        result = await db.execute(select(Speaker).where(Speaker.speaker_id == call.speaker_id))
        speaker = result.scalar_one_or_none()
        if speaker is None or not speaker.reference_embedding:
            return None

        return speaker.reference_embedding
    except Exception:
        logger.exception("Failed to load speaker reference for call %s", call_id)
        return None


async def _process_chunk(
    chunk: AudioChunk,
    call_id: str,
    reference_embedding: Optional[list],
    context: ContextFactors,
    weights: RiskWeights,
    db: AsyncSession,
    t_preprocessing_start: float,
    t_chunk_ready: float,
    t_message_received: float,
) -> dict:
    """
    Runs the full per-chunk pipeline (synthetic voice inference ->
    speaker verification -> prosody analysis -> risk engine -> temporal
    smoothing) and returns the JSON-serializable result dict, including
    measured latency breakdowns.
    """
    t_inference_start = time.perf_counter()

    # --- synthetic voice inference ---
    detection = detection_service.analyze_chunk(chunk)

    # --- speaker verification (only if a reference embedding is available) ---
    speaker_similarity: Optional[float] = None
    identity_match: Optional[bool] = None
    if reference_embedding:
        try:
            verification = speaker_service.verify_chunk(
                reference_embedding, chunk, threshold=settings.SPEAKER_SIMILARITY_THRESHOLD
            )
            speaker_similarity = verification.speaker_similarity
            identity_match = verification.identity_match
        except Exception:
            logger.exception("Speaker verification failed for call %s; continuing without it", call_id)

    # --- prosody analysis ---
    prosody = prosody_service.analyze_chunk(chunk)

    t_inference_end = time.perf_counter()

    # --- risk engine (raw, per-chunk, all 10 signals) ---
    raw_risk = compute_risk(
        synthetic_probability=detection.synthetic_probability,
        speaker_similarity=speaker_similarity,
        prosody_anomaly=prosody.prosody_anomaly,
        context=context,
        weights=weights,
    )

    # --- temporal smoothing (applied to the final risk score) ---
    smoothed_score = smoothing_registry.get(f"{call_id}:risk").update(raw_risk.risk_score)
    smoothed_score = round(min(100.0, max(0.0, smoothed_score)), 1)
    smoothed_level = score_to_level(smoothed_score)
    smoothed_action = determine_recommended_action(smoothed_level, context)

    alert = maybe_generate_alert(
        call_id,
        RiskResult(
            risk_score=smoothed_score,
            risk_level=smoothed_level,
            reasons=raw_risk.reasons,
            recommended_action=smoothed_action,
        ),
    )

    context_risk_summary = compute_context_risk_summary(context, weights)
    t_end = time.perf_counter()

    result = {
        "type": "result",
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_start_time": round(chunk.start_time, 3),
        "chunk_end_time": round(chunk.end_time, 3),
        "synthetic_probability": round(detection.synthetic_probability, 4),
        "speaker_similarity": round(speaker_similarity, 4) if speaker_similarity is not None else None,
        "identity_match": identity_match,
        "prosody_anomaly": round(prosody.prosody_anomaly, 4),
        "context_risk": round(context_risk_summary, 4),
        "risk_score": smoothed_score,
        "risk_level": smoothed_level,
        "reasons": raw_risk.reasons,
        "recommended_action": smoothed_action,
        "alert": alert.model_dump(mode="json") if alert else None,
        "model_version": detection.model_version,
        "disclaimer": RISK_DISCLAIMER,
        "latency_ms": {
            "preprocessing": round((t_chunk_ready - t_preprocessing_start) * 1000, 2),
            "inference": round((t_inference_end - t_inference_start) * 1000, 2),
            "total": round((t_end - t_message_received) * 1000, 2),
        },
    }

    # Persist a risk event so call history (GET /api/v1/calls/{id}/risk)
    # and the dashboard can show WHY a call received its score, even
    # after the live connection ends. A failure here (e.g. the call_id
    # was never created via POST /api/v1/calls, so the foreign key has
    # no match) must never break the live stream -- log and continue.
    try:
        db.add(
            RiskEvent(
                call_id=call_id,
                synthetic_probability=detection.synthetic_probability,
                speaker_similarity=speaker_similarity,
                prosody_anomaly=prosody.prosody_anomaly,
                context_risk=context_risk_summary,
                risk_score=smoothed_score,
                risk_level=smoothed_level,
                reasons=raw_risk.reasons,
                recommended_action=smoothed_action,
                model_version=detection.model_version,
            )
        )
        await db.commit()
    except Exception:
        logger.warning("Could not persist risk event for call %s (call may not exist in DB)", call_id)
        await db.rollback()

    return result


@router.websocket("/ws/calls/{call_id}/audio")
async def call_audio_stream(websocket: WebSocket, call_id: str, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    preprocessor = StreamingPreprocessor()
    weights = weights_from_settings()
    context = ContextFactors()

    reference_embedding = await _load_reference_embedding(db, call_id)
    audit_log("ws_call_connected", call_id=call_id, speaker_reference_loaded=bool(reference_embedding))

    await websocket.send_json(
        {
            "type": "connected",
            "call_id": call_id,
            "speaker_reference_loaded": bool(reference_embedding),
            "disclaimer": RISK_DISCLAIMER,
        }
    )

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError:
                # receive() raises RuntimeError if called after a disconnect
                # message has already been processed -- treat as a clean end.
                break

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                # Mid-call context updates (e.g. "transaction requested",
                # "bypass requested") arrive as JSON text frames.
                try:
                    payload = json.loads(message["text"])
                    context = ContextFactors(**{**context.model_dump(), **payload})
                    await websocket.send_json({"type": "context_updated", "call_id": call_id})
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    await websocket.send_json(
                        {"type": "error", "error": f"Invalid context update payload: {exc}"}
                    )
                continue

            raw_bytes = message.get("bytes")
            if not raw_bytes:
                continue

            t_message_received = time.perf_counter()

            try:
                chunk_iter = preprocessor.push(raw_bytes)
                while True:
                    t_preprocessing_start = time.perf_counter()
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        break
                    t_chunk_ready = time.perf_counter()

                    if not chunk.is_voiced:
                        # VAD says this window has no meaningful speech --
                        # skip inference rather than scoring silence.
                        continue

                    try:
                        result = await _process_chunk(
                            chunk=chunk,
                            call_id=call_id,
                            reference_embedding=reference_embedding,
                            context=context,
                            weights=weights,
                            db=db,
                            t_preprocessing_start=t_preprocessing_start,
                            t_chunk_ready=t_chunk_ready,
                            t_message_received=t_message_received,
                        )
                        await websocket.send_json(result)
                    except Exception:
                        # A failure analyzing one chunk must not take down
                        # the whole stream -- report it and keep going.
                        logger.exception(
                            "Error analyzing audio chunk for call %s; continuing stream", call_id
                        )
                        await websocket.send_json(
                            {
                                "type": "error",
                                "call_id": call_id,
                                "error": "Failed to analyze this audio chunk; continuing stream.",
                            }
                        )

            except WebSocketDisconnect:
                raise
            except Exception as exc:
                # Corrupt/unsupported audio, decode failures, etc. -- the
                # connection stays open and the client can keep sending
                # valid chunks.
                logger.warning("Invalid/corrupt audio chunk on call %s: %s", call_id, exc)
                await websocket.send_json(
                    {
                        "type": "error",
                        "call_id": call_id,
                        "error": "Could not decode audio chunk (corrupt or unsupported format).",
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unexpected error in WebSocket stream for call %s", call_id)
    finally:
        smoothing_registry.reset(f"{call_id}:risk")
        audit_log("ws_call_disconnected", call_id=call_id)

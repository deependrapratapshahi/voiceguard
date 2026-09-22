import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.entities import Call, RiskEvent
from app.models.schemas import CallCreateRequest, CallResponse, CallRiskSnapshot

router = APIRouter(prefix="/api/v1/calls", tags=["calls"])


@router.post("", response_model=CallResponse)
async def create_call(payload: CallCreateRequest, db: AsyncSession = Depends(get_db)):
    call = Call(
        call_id=f"CALL-{uuid.uuid4().hex[:8].upper()}",
        caller_label=payload.caller_label,
        speaker_id=payload.speaker_id,
        status="active",
        started_at=datetime.now(timezone.utc),
        is_demo=payload.is_demo,
    )
    db.add(call)
    await db.commit()
    await db.refresh(call)
    return call


@router.get("", response_model=List[CallResponse])
async def list_calls(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Call).order_by(Call.started_at.desc()).limit(100))
    return result.scalars().all()


@router.get("/{call_id}", response_model=CallResponse)
async def get_call(call_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Call).where(Call.call_id == call_id))
    call = result.scalar_one_or_none()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@router.get("/{call_id}/risk", response_model=List[CallRiskSnapshot])
async def get_call_risk_history(call_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(RiskEvent).where(RiskEvent.call_id == call_id).order_by(RiskEvent.timestamp.asc())
    )
    events = result.scalars().all()
    return [
        CallRiskSnapshot(
            call_id=call_id,
            timestamp=e.timestamp,
            synthetic_probability=e.synthetic_probability,
            speaker_similarity=e.speaker_similarity,
            prosody_anomaly=e.prosody_anomaly,
            risk_score=e.risk_score,
            risk_level=e.risk_level,
            reasons=e.reasons or [],
            recommended_action=e.recommended_action,
        )
        for e in events
    ]

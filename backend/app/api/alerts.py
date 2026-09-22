from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.entities import Alert
from app.models.schemas import AlertCreateRequest, AlertResponse

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
async def list_alerts(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).order_by(Alert.created_at.desc()).limit(200))
    return result.scalars().all()


@router.post("", response_model=AlertResponse)
async def create_alert(payload: AlertCreateRequest, db: AsyncSession = Depends(get_db)):
    import uuid

    alert = Alert(
        alert_id=f"ALERT-{uuid.uuid4().hex[:8].upper()}",
        call_id=payload.call_id,
        severity=payload.severity,
        message=payload.message,
        recommended_action=payload.recommended_action,
        status="open",
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert

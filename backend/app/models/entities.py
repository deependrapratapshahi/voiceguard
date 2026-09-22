"""
ORM entities.

Per the privacy design (see docs/privacy.md), raw audio is NOT stored in
any of these tables by default -- only derived metadata, scores, and
identifiers.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    call_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    caller_label: Mapped[str] = mapped_column(String(128), nullable=True)
    speaker_id: Mapped[str] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active|ended
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True)

    risk_events: Mapped[list["RiskEvent"]] = relationship(back_populates="call")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="call")


class Speaker(Base):
    __tablename__ = "speakers"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    speaker_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128))
    embedding_model_version: Mapped[str] = mapped_column(String(32))
    # Reference embedding is stored as a JSON-encoded float vector, not raw audio.
    reference_embedding: Mapped[list] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    synthetic_probability: Mapped[float] = mapped_column(Float, nullable=True)
    speaker_similarity: Mapped[float] = mapped_column(Float, nullable=True)
    prosody_anomaly: Mapped[float] = mapped_column(Float, nullable=True)
    context_risk: Mapped[float] = mapped_column(Float, nullable=True)

    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    reasons: Mapped[list] = mapped_column(JSONB, default=list)
    recommended_action: Mapped[str] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str] = mapped_column(String(32))

    call: Mapped["Call"] = relationship(back_populates="risk_events")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    alert_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    recommended_action: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="open")  # open|acknowledged|resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    call: Mapped["Call"] = relationship(back_populates="alerts")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    component: Mapped[str] = mapped_column(String(32))  # deepfake|speaker|prosody
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    event_type: Mapped[str] = mapped_column(String(64))
    call_id: Mapped[str] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

"""
Privacy service abstraction.

Centralizes privacy-relevant decisions so they are configured in one
place rather than scattered through the codebase:
- whether raw audio may be buffered at all, and for how long
- whether logging should be feature-only (no raw audio, no transcripts)
- deletion controls for stored artifacts

This module does NOT make any legal-compliance claims. See
docs/privacy.md: organizations deploying VOICEGUARD must perform their
own legal/regulatory assessment (e.g. wiretapping consent laws, data
protection regulation) for their jurisdiction and use case.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import get_settings
from app.utils.logging import audit_log

settings = get_settings()


@dataclass
class RetentionDecision:
    allow_raw_buffer: bool
    max_buffer_seconds: int
    feature_only_logging: bool


def get_retention_policy() -> RetentionDecision:
    return RetentionDecision(
        allow_raw_buffer=settings.RAW_AUDIO_RETENTION_SECONDS > 0,
        max_buffer_seconds=settings.RAW_AUDIO_RETENTION_SECONDS,
        feature_only_logging=settings.FEATURE_ONLY_LOGGING,
    )


class EphemeralAudioBuffer:
    """
    Optional short-lived in-memory buffer for raw audio, only active if
    RAW_AUDIO_RETENTION_SECONDS > 0. Never persists to disk. Entries are
    dropped once they exceed the configured retention window.
    """

    def __init__(self):
        self._entries: dict[str, tuple[bytes, float]] = {}

    def put(self, key: str, data: bytes) -> None:
        policy = get_retention_policy()
        if not policy.allow_raw_buffer:
            return
        self._entries[key] = (data, time.time())

    def get(self, key: str) -> bytes | None:
        self._purge_expired()
        entry = self._entries.get(key)
        return entry[0] if entry else None

    def delete(self, key: str) -> None:
        self._entries.pop(key, None)
        audit_log("raw_audio_deleted", key=key)

    def _purge_expired(self) -> None:
        policy = get_retention_policy()
        now = time.time()
        expired = [
            k for k, (_, ts) in self._entries.items()
            if now - ts > policy.max_buffer_seconds
        ]
        for k in expired:
            self.delete(k)


ephemeral_audio_buffer = EphemeralAudioBuffer()

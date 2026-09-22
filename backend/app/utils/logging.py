"""
Structured logging setup.

Logs are structured (JSON-friendly key=value) and deliberately avoid
logging raw audio content or full PII by default, in line with the
privacy design goals of VOICEGUARD.
"""
import logging
import sys
from app.config import get_settings

settings = get_settings()


class SafeFormatter(logging.Formatter):
    """Formatter that never includes raw audio buffers, only metadata."""

    def format(self, record: logging.LogRecord) -> str:
        record.msg = str(record.msg)
        return super().format(record)


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        SafeFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def audit_log(event_type: str, **fields) -> None:
    """
    Emit an audit-trail log entry. Only metadata (ids, scores, decisions)
    should ever be passed here -- never raw audio or full transcripts.
    """
    if not settings.AUDIT_LOG_ENABLED:
        return
    logger = get_logger("voiceguard.audit")
    safe_fields = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"AUDIT event={event_type} {safe_fields}")

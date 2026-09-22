"""
Security utilities.

This module is "auth-ready": it provides JWT and API-key verification
scaffolding so real authentication can be enabled without changing any
API route signatures. In demo mode, authentication can be disabled via
the DEMO_MODE flag in config for local development only.

No offensive-security functionality is implemented in this module.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Header, HTTPException, status
from jose import JWTError, jwt

from app.config import get_settings

settings = get_settings()


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.API_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.API_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from exc


async def get_optional_api_key(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    """
    Placeholder dependency for routes that will require an API key once
    authentication is switched on for a deployment. Returns None in demo
    mode rather than rejecting requests, so the prototype remains usable
    out of the box.
    """
    return x_api_key


def safe_error_message(exc: Exception) -> str:
    """
    Convert internal exceptions into a message safe to return to clients,
    without leaking stack traces, file paths, or internal configuration.
    """
    return "An internal error occurred while processing the request."

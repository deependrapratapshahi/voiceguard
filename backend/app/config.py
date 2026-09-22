"""
Application configuration.

All configuration is loaded from environment variables (see .env.example).
No secrets or environment-specific values are hard-coded here.
"""
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General ---
    APP_NAME: str = "VOICEGUARD"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://voiceguard:voiceguard@localhost:5432/voiceguard"

    # --- Redis (streaming state / temporal smoothing buffers) ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Security ---
    API_SECRET: str = "change-me-in-.env"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: str = "http://localhost:5173"

    # --- ML / model management ---
    MODEL_PATH: str = "./ml_artifacts"
    DEEPFAKE_MODEL_VERSION: str = "cnn-baseline-v1"
    SPEAKER_MODEL_VERSION: str = "baseline-v1"
    PROSODY_MODEL_VERSION: str = "baseline-v1"
    USE_PRETRAINED_ENCODERS: bool = False  # falls back to classical baseline if False/unavailable
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.55

    # --- Audio pipeline ---
    TARGET_SAMPLE_RATE: int = 16000
    CHUNK_DURATION_SECONDS: float = 2.0
    CHUNK_OVERLAP_SECONDS: float = 0.5
    MAX_UPLOAD_MB: int = 25

    # --- Privacy ---
    RAW_AUDIO_RETENTION_SECONDS: int = 0  # 0 = never persist raw audio to disk
    FEATURE_ONLY_LOGGING: bool = True
    AUDIT_LOG_ENABLED: bool = True

    # --- Risk engine defaults (overridable via API) ---
    SYNTHETIC_WEIGHT: float = 0.25
    SPEAKER_WEIGHT: float = 0.15
    PROSODY_WEIGHT: float = 0.08
    CALLER_REPUTATION_WEIGHT: float = 0.10
    REGISTERED_CONTACT_WEIGHT: float = 0.07
    TRANSACTION_WEIGHT: float = 0.15
    PRIVILEGED_ACTION_WEIGHT: float = 0.06
    URGENCY_WEIGHT: float = 0.05
    BYPASS_CALLBACK_WEIGHT: float = 0.06
    HISTORICAL_FRAUD_WEIGHT: float = 0.03

    # --- Risk level thresholds (score <= threshold maps to that level) ---
    RISK_THRESHOLD_LOW: float = 30.0
    RISK_THRESHOLD_MEDIUM: float = 60.0
    RISK_THRESHOLD_HIGH: float = 80.0  # above this -> CRITICAL

    # --- Contextual factor scaling ---
    TRANSACTION_LOW_THRESHOLD: float = 1000.0   # below this, transaction contributes ~0 risk
    TRANSACTION_HIGH_THRESHOLD: float = 50000.0  # at/above this, transaction contributes full risk
    HISTORICAL_FRAUD_CAP: int = 5  # flag count at which historical-fraud risk maxes out
    SPEAKER_UNCERTAIN_LOW: float = 0.35  # below this similarity -> "low similarity" reason
    SPEAKER_UNCERTAIN_HIGH: float = 0.65  # between low/high -> "uncertain" reason

    # --- Temporal smoothing ---
    SMOOTHING_STRATEGY: str = "ema"  # "moving_average" | "ema"
    EMA_ALPHA: float = 0.4
    MOVING_AVERAGE_WINDOW: int = 5

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

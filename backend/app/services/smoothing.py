"""
Temporal smoothing for streaming risk/detection scores.

Individual chunk-level predictions fluctuate; this module stabilizes
them via a configurable moving average or exponential moving average
(EMA), keyed per call_id so multiple concurrent calls don't interfere.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config import get_settings

settings = get_settings()


@dataclass
class SmoothingState:
    strategy: str
    window: deque = field(default_factory=deque)
    ema_value: float | None = None

    def update(self, value: float) -> float:
        if self.strategy == "moving_average":
            self.window.append(value)
            while len(self.window) > settings.MOVING_AVERAGE_WINDOW:
                self.window.popleft()
            return sum(self.window) / len(self.window)

        # exponential moving average
        alpha = settings.EMA_ALPHA
        self.ema_value = value if self.ema_value is None else alpha * value + (1 - alpha) * self.ema_value
        return self.ema_value


class SmoothingRegistry:
    """In-memory per-call smoothing state. For multi-worker deployments,
    back this with Redis instead (see app/config.py REDIS_URL)."""

    def __init__(self):
        self._states: dict[str, SmoothingState] = {}

    def get(self, call_id: str) -> SmoothingState:
        if call_id not in self._states:
            self._states[call_id] = SmoothingState(strategy=settings.SMOOTHING_STRATEGY)
        return self._states[call_id]

    def reset(self, call_id: str) -> None:
        self._states.pop(call_id, None)


smoothing_registry = SmoothingRegistry()

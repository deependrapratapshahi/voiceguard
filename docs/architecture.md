# Architecture

## Overview

VOICEGUARD processes audio through a linear pipeline with four parallel analysis branches that
converge into a single risk score:

```
Audio source → Ingestion → VAD → Preprocessing
                                       │
        ┌──────────────────────────────────────────────┐
        │              Parallel analysis                │
        │  1. Synthetic voice detector                   │
        │  2. Speaker verification                        │
        │  3. Prosody / acoustic analysis                  │
        │  4. Contextual risk analysis                      │
        └──────────────────────────────────────────────┘
                                       │
                     Feature/result aggregation
                                       │
                            Temporal smoothing
                                       │
                            Risk scoring engine
                                       │
                              Alert engine
                                       │
                       REST / WebSocket API → React dashboard
```

## Components

### Audio ingestion & preprocessing (`app/services/audio_service.py`)
Decodes uploaded or streamed audio, resamples to 16kHz mono, normalizes amplitude, removes
silence, and chunks into overlapping ~1-3s windows. The `StreamingPreprocessor` class buffers
incoming WebSocket byte frames and yields chunks incrementally — the system never assumes the
whole call is available before predicting.

### Synthetic voice detector (`ml/deepfake/`)
`model.py` defines a `BaselineClassicalDetector` (logistic regression over summarized log-mel +
MFCC features) and a `PretrainedEncoderDetector` interface for swapping in wav2vec2/HuBERT/WavLM
later. `inference.py` is the single entry point the backend calls; it tries the pretrained path
only if enabled and available, and always falls back to the baseline rather than crashing.

### Speaker verification (`ml/speaker/`)
`embeddings.py` extracts a baseline embedding (MFCC + pitch statistics); `verification.py`
computes cosine similarity against a stored reference and applies a configurable decision
threshold, with a calibration helper for tuning against labeled genuine/impostor score pairs.

### Prosody analysis (`ml/prosody/`)
Extracts pitch, jitter, shimmer, pause statistics, and spectral features, then scores deviation
from configurable "expected" ranges for natural speech into a 0-1 anomaly score.

### Contextual fraud-risk engine (`app/services/risk_engine.py`)
Combines ten signals into a single, explainable 0-100 score: synthetic speech probability,
speaker similarity, prosody anomaly, caller reputation, registered-contact status, transaction
amount, privileged action, urgency, a request to bypass registered callback verification, and a
historical fraud indicator. Each signal is converted to a normalized risk fraction, multiplied by
a configurable weight (`RiskWeights` / environment variables), and summed. Every contributing
signal above its notable threshold produces a specific, human-readable reason — the score is
never a black box. LOW/MEDIUM/HIGH/CRITICAL thresholds are configurable via settings, and the
recommended action (`ALLOW`, `MONITOR`, `SECONDARY_VERIFICATION`, `CALLBACK_AND_MFA`, `ESCALATE`)
depends on both the level and which specific factors triggered (e.g. a HIGH-risk call with a
callback-bypass request routes to `CALLBACK_AND_MFA` rather than generic
`SECONDARY_VERIFICATION`).

### Temporal smoothing (`app/services/smoothing.py`)
Applies EMA or moving-average smoothing per call to stabilize chunk-level fluctuations before
they reach the risk engine, keyed by `call_id` (swap the in-memory registry for Redis in a
multi-worker deployment).

### Alert engine (`app/services/alert_service.py`)
Generates alert objects when risk crosses configured thresholds (MEDIUM/HIGH/CRITICAL);
structured so email/SMS/push channels can be added later without touching callers.

### APIs (`app/api/`, `app/websocket/`)
REST endpoints for calls, detection, speakers, risk evaluation, alerts, and model info; a
WebSocket endpoint (`/ws/calls/{call_id}/audio`) for near-real-time streaming analysis.

### Frontend (`frontend/src/`)
A SOC-style React dashboard: overview stats, live call monitoring with a real-time risk chart,
call detail timelines, a speaker registry, an alerts table, analytics, and a fully local demo
mode.

## Data flow guarantees

- No component assumes the full audio is available up front.
- Every model output is bounded (0-1 probability/score, or 0-100 for the final risk score) and
  labeled with a `model_version` so baseline vs. pretrained outputs are distinguishable.
- Raw audio is never written to the database; only derived scores/metadata are persisted (see
  `docs/privacy.md`).

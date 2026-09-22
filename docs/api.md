# API Documentation

Base URL (local): `http://localhost:8000`
Interactive OpenAPI docs: `http://localhost:8000/docs`

All endpoints are versioned under `/api/v1`. All probabilistic fields (`synthetic_probability`,
`speaker_similarity`, `prosody_anomaly`, `confidence`) are floats in `[0, 1]`. `risk_score` is a
float in `[0, 100]`. None of these values should be interpreted as certainty.

## Health

### `GET /api/v1/health`
Returns service status plus live database/Redis connectivity checks.

```json
{
  "status": "ok",
  "app": "VOICEGUARD",
  "env": "development",
  "database": "connected",
  "redis": "connected"
}
```

## Calls

### `POST /api/v1/calls`
Create a call record.
Body: `{ "caller_label": "string?", "speaker_id": "string?", "is_demo": true }`
Returns: `CallResponse`

### `GET /api/v1/calls`
List the most recent 100 calls.

### `GET /api/v1/calls/{call_id}`
Fetch a single call by its `call_id` (e.g. `CALL-A1B2C3D4`).

### `GET /api/v1/calls/{call_id}/risk`
Returns the ordered list of risk snapshots recorded for a call.

## Detection

### `POST /api/v1/detection/analyze`
Multipart upload (`file`) — runs the full offline pipeline (preprocessing → chunking →
detection → prosody) over an uploaded audio file. Accepts `.wav`, `.mp3`, `.flac`, `.ogg`,
`.m4a` up to `MAX_UPLOAD_MB` (default 25MB).

```json
{
  "num_chunks_analyzed": 4,
  "average_synthetic_probability": 0.84,
  "average_prosody_anomaly": 0.64,
  "per_chunk_detection": [{"synthetic_probability": 0.84, "label": "LIKELY_SYNTHETIC", "model_version": "baseline-v1"}],
  "per_chunk_prosody": [{"prosody_anomaly": 0.64, "features": { "...": "..." }}]
}
```

## Speakers

### `POST /api/v1/speakers?speaker_id=...&display_name=...`
Multipart upload (`file`) of reference audio — builds and stores a reference embedding for a
speaker profile.

### `POST /api/v1/speakers/{speaker_id}/verify`
Multipart upload (`file`) of candidate audio — compares against the stored reference embedding.

```json
{ "speaker_similarity": 0.81, "identity_match": true, "confidence": 0.79 }
```

## Risk

### `POST /api/v1/risk/evaluate`
Query params: `synthetic_probability`, `speaker_similarity`, `prosody_anomaly` (all optional
floats). Body: `ContextFactors` — the ten combined signals are synthetic speech probability,
speaker similarity, and prosody anomaly (query params above) plus these contextual factors in
the request body:

```json
{
  "caller_reputation_score": 0.2,
  "caller_is_registered_contact": false,
  "is_new_device": true,
  "transaction_value": 75000,
  "urgency_indicated": true,
  "bypass_requested": true,
  "privileged_operation": true,
  "unusual_call_time": false,
  "historical_fraud_flags": 2
}
```

Response:
```json
{
  "risk_score": 87.3,
  "risk_level": "CRITICAL",
  "reasons": [
    "High synthetic speech probability",
    "Low speaker similarity to reference voice",
    "Caller has a poor reputation score",
    "Caller is not a registered contact",
    "High-value transaction",
    "Request involves a privileged operation",
    "Urgency indicators present in the request",
    "Caller requested bypass of registered callback",
    "Prior fraud indicators on record for this caller/account"
  ],
  "recommended_action": "ESCALATE"
}
```

`recommended_action` is one of `ALLOW`, `MONITOR`, `SECONDARY_VERIFICATION`,
`CALLBACK_AND_MFA`, or `ESCALATE`. Weights for each of the ten signals and the
LOW/MEDIUM/HIGH/CRITICAL score thresholds are configured server-side (see `.env` /
`app/config.py`) — see [`docs/ml_pipeline.md`](ml_pipeline.md) is for models; for the risk
engine itself see the `app/services/risk_engine.py` module docstring, which lists every
configurable weight and threshold.

## Alerts

### `GET /api/v1/alerts`
List the most recent 200 alerts.

### `POST /api/v1/alerts`
Create an alert manually. Body: `{ "call_id", "severity", "message", "recommended_action" }`.

## Models

### `GET /api/v1/models`
Returns the active model name/version per component (deepfake, speaker, prosody) and whether
each is a baseline or pretrained model.

## WebSocket

### `WS /ws/calls/{call_id}/audio`

Pipeline per chunk: audio decoding → preprocessing → VAD → synthetic voice inference → speaker
verification → prosody analysis → risk engine → temporal smoothing (applied to the final risk
score for stability). A JSON message is sent back after every successfully processed, voiced
chunk. Invalid/corrupt audio never closes the connection — it produces an `error` message and
the stream continues.

**On connect**, the server immediately sends a handshake:
```json
{
  "type": "connected",
  "call_id": "CALL-001",
  "speaker_reference_loaded": true,
  "disclaimer": "This is a probabilistic risk indicator... not proof of fraud..."
}
```
`speaker_reference_loaded` is `true` only if the call was created with a `speaker_id` that has a
registered reference embedding (via `POST /api/v1/speakers`) — otherwise speaker verification is
skipped for that call and `speaker_similarity` will be `null` in every result.

**Client → server:**
- Binary frames: self-contained audio chunks (e.g. small WAV files). Each frame is decoded
  independently, then concatenated into an internal streaming buffer.
- Text frames: JSON to update context factors mid-call, e.g.
  `{"bypass_requested": true, "transaction_value": 50000}`. Acknowledged with
  `{"type": "context_updated", "call_id": "..."}`, or `{"type": "error", "error": "..."}` if the
  payload is invalid JSON.

**Server → client, per voiced chunk:**
```json
{
  "type": "result",
  "call_id": "CALL-001",
  "timestamp": "2026-08-28T10:00:00Z",
  "chunk_start_time": 0.0,
  "chunk_end_time": 2.0,
  "synthetic_probability": 0.82,
  "speaker_similarity": 0.64,
  "identity_match": false,
  "prosody_anomaly": 0.71,
  "context_risk": 0.45,
  "risk_score": 84.0,
  "risk_level": "CRITICAL",
  "reasons": ["High synthetic speech probability", "Low speaker similarity to reference voice"],
  "recommended_action": "CALLBACK_AND_MFA",
  "alert": { "...": "..." },
  "model_version": "cnn-baseline-v1",
  "disclaimer": "This is a probabilistic risk indicator... not proof of fraud...",
  "latency_ms": { "preprocessing": 3.2, "inference": 41.7, "total": 46.1 }
}
```

`speaker_similarity` and `identity_match` are `null` whenever no speaker reference is loaded for
the call. `risk_score`/`risk_level`/`recommended_action` reflect the **temporally smoothed**
score; `reasons` explain the current chunk's raw signal readings. `latency_ms` values are
measured with `time.perf_counter()`, never fabricated.

**On invalid/corrupt audio:**
```json
{ "type": "error", "call_id": "CALL-001", "error": "Could not decode audio chunk (corrupt or unsupported format)." }
```
The connection stays open — send a valid chunk next and the stream resumes normally.

**Disconnection:** closing the connection (client or server side) is handled cleanly; per-call
temporal-smoothing state is reset so a reconnect with the same `call_id` starts fresh.

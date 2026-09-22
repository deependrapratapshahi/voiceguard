# Deployment

## Local development

```bash
cp .env.example .env
make install
make backend    # terminal 1
make frontend   # terminal 2
```

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Services: `postgres` (5432), `redis` (6379), `backend` (8000), `frontend` (5173). The backend
waits for Postgres/Redis health checks before starting. ML dependencies (`torch`,
`transformers`) are attempted first; if unavailable in the build environment the image falls
back to installing a lighter `requirements.lite.txt` set (baseline mode still works fully).

## Database migrations

For anything beyond local development, replace the `init_models()` auto-create-on-startup
behavior with [Alembic](https://alembic.sqlalchemy.org/) migrations:

```bash
pip install alembic
alembic init backend/migrations
# configure alembic.ini's sqlalchemy.url from DATABASE_URL
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

## Security hardening checklist for non-local deployments

- [ ] Set a strong, random `API_SECRET` (never reuse the `.env.example` placeholder).
- [ ] Put real authentication behind the JWT-ready scaffolding in `app/utils/security.py` — wire
      `get_optional_api_key`/`decode_access_token` into route dependencies as required auth.
- [ ] Restrict `CORS_ORIGINS` to your actual frontend origin(s) — never `*` in production.
- [ ] Add a rate-limiting layer (e.g. `slowapi` or an API gateway) in front of the REST/WebSocket
      endpoints; the codebase leaves a clear seam for this but does not include a specific
      implementation.
- [ ] Terminate TLS in front of the backend (reverse proxy / load balancer) — do not expose
      Uvicorn directly to the internet.
- [ ] Set `RAW_AUDIO_RETENTION_SECONDS` deliberately after your privacy/legal review (see
      `docs/privacy.md`); default is `0` (no raw-audio buffering).
- [ ] Run vulnerability scanning on the Docker images and pinned dependency versions before
      shipping.
- [ ] Review `docs/privacy.md` for consent/compliance considerations specific to your
      jurisdiction and use case.

## Scaling notes

- The in-memory `SmoothingRegistry` (`app/services/smoothing.py`) and `EphemeralAudioBuffer`
  (`app/services/privacy_service.py`) are per-process. For multi-worker/multi-instance
  deployments, back these with Redis (`REDIS_URL` is already configured and available) instead
  of the in-memory dict, so all workers/instances share consistent state for a given call.
- WebSocket connections are stateful; if you scale horizontally, use sticky sessions or a shared
  state layer (Redis) so a call's stream stays consistent across reconnects.

## Latency expectations

The dashboard surfaces measured (not fabricated) preprocessing, inference, and end-to-end
latency. If the deployment target cannot sustain true real-time inference (e.g. CPU-only
hardware with the pretrained-encoder path enabled), label the deployment clearly as "simulation
mode" per the project's quality requirements rather than presenting simulated timing as live
performance.

# Privacy Considerations

**VOICEGUARD does not, and cannot, make any claims of legal or regulatory compliance.**
Organizations deploying this system are responsible for their own legal/compliance assessment —
including call-recording consent laws (which vary by jurisdiction and often require one-party or
all-party consent), biometric data regulations (e.g. GDPR biometric categories, BIPA in Illinois),
and any sector-specific rules (e.g. financial services, telecom). Nothing in this document or the
codebase constitutes legal advice.

## Design principles

1. **Raw audio is not stored by default.** `RAW_AUDIO_RETENTION_SECONDS=0` means the ephemeral
   in-memory buffer (`app/services/privacy_service.py:EphemeralAudioBuffer`) never actually
   buffers anything — audio is processed in memory, chunk by chunk, and discarded immediately
   after feature extraction.
2. **Feature-only logging.** With `FEATURE_ONLY_LOGGING=true` (default), logs and audit trails
   contain only metadata — call IDs, scores, decisions, timestamps — never raw audio bytes or
   transcripts (see `app/utils/logging.py:audit_log`, which only accepts keyword metadata).
3. **Database contains no raw audio.** The `calls`, `risk_events`, `alerts`, and
   `security_events` tables store only IDs, scores, labels, and reasons (see
   `app/models/entities.py`). Speaker reference embeddings are stored as numeric vectors, not
   audio — they cannot be trivially converted back into intelligible speech.
4. **Configurable retention.** If a deployment needs short-term raw-audio buffering (e.g. for
   human review of a flagged call), set `RAW_AUDIO_RETENTION_SECONDS` to a positive value; the
   `EphemeralAudioBuffer` will then hold data in memory only, automatically purging entries older
   than the configured window, and logs every deletion via `audit_log`.
5. **Deletion controls.** `EphemeralAudioBuffer.delete()` removes a buffered entry immediately
   and records an audit event. No raw-audio persistence to disk exists in this prototype.
6. **Environment-variable configuration.** All privacy-relevant behavior (`RAW_AUDIO_RETENTION_SECONDS`,
   `FEATURE_ONLY_LOGGING`, `AUDIT_LOG_ENABLED`) is controlled via `.env`, not hard-coded, so a
   deployment can tune it to its own policy and legal review.
7. **Audit logging.** Every WebSocket connect/disconnect and privacy-relevant action (e.g. raw
   audio deletion) is recorded via `audit_log`, itself gated by `AUDIT_LOG_ENABLED`.

## Minimizing personal information

- Speaker profiles store a `speaker_id` and `display_name` you choose — use pseudonymous IDs in
  any non-demo deployment if display names are not operationally necessary.
- Call records store an optional `caller_label` — avoid putting personally identifiable
  information there unless your deployment's data-handling policy explicitly allows it.

## What this project does NOT do

- It does not encrypt data at rest by default (add disk/DB-level encryption at the
  infrastructure layer for production).
- It does not implement consent-management or wiretap-notification workflows — those must be
  built into whatever telecom/IVR integration calls into VOICEGUARD's APIs.
- It does not certify GDPR/CCPA/BIPA/HIPAA compliance or any other regulatory framework.

## Recommendation

Before any production or pilot deployment involving real callers, have qualified legal counsel
review: (1) applicable call-recording/consent laws in every jurisdiction you operate in, (2)
biometric data handling requirements for the speaker-embedding feature, and (3) your data
retention and deletion policy against `RAW_AUDIO_RETENTION_SECONDS` and the database retention
policy you configure.

# ML Pipeline

## Design principles

1. **No fabricated inference.** Whenever a real (trained or pretrained) model is unavailable,
   the system uses an explicitly named, clearly labeled baseline rather than pretending to run
   a stronger model.
2. **Probabilities, not certainties.** All outputs are framed as `synthetic_probability`,
   `speaker_similarity`, `prosody_anomaly`, or `risk_score` — never "real"/"fake" absolutes.
3. **Swappable architectures.** Every model sits behind a stable `inference.py` entry point so
   the API/frontend contract never changes when the underlying model is upgraded.

## Synthetic voice detection (`ml/deepfake/`)

- **Baseline** (`model.py:BaselineClassicalDetector`): logistic regression over a feature vector
  built from summarized (mean+std) log-mel spectrogram and MFCC statistics
  (`ml/common/audio.py:extract_baseline_feature_vector`). No GPU or downloads required. If no
  trained weights exist yet, it returns `UNCERTAIN` with `model_version` suffixed
  `-untrained` rather than fabricating a confident score.
- **Optional pretrained path** (`model.py:PretrainedEncoderDetector`): interface for a
  self-supervised speech encoder (wav2vec2 / HuBERT / WavLM) with a lightweight classification
  head. Only used when `USE_PRETRAINED_ENCODERS=true` **and** local weights are found; any
  failure (missing deps, missing weights, runtime error) transparently falls back to the
  baseline (`inference.py`).
- **Training** (`train.py`): loads `datasets/real/` + `datasets/synthetic/`, performs a
  **speaker-independent** train/val/test split (`dataset.py:speaker_independent_split`) to avoid
  the same speaker leaking across splits, reports cross-validation accuracy, and saves the
  fitted model to `ml_artifacts/deepfake_baseline.pkl`.
- **Evaluation** (`evaluate.py`): reports accuracy, precision/recall, and Equal Error Rate (EER)
  — the standard anti-spoofing metric — on the held-out speaker-independent test split. Prints
  an explicit message rather than fake numbers if no model or test data exists.

### Dataset expectations
Directory-based:
```
datasets/real/<file>.wav
datasets/synthetic/<file>.wav
```
or metadata-driven (CSV/JSON with `file_path`, `label`, `speaker_id` columns — see
`dataset.py:load_metadata_dataset`). VOICEGUARD does **not** download any dataset automatically;
obtain e.g. a licensed subset of [ASVspoof](https://www.asvspoof.org) yourself and place it
under `datasets/`, then run `scripts/validate_dataset.py` to check structure, class balance, and
file integrity before training.

## Speaker verification (`ml/speaker/`)

- **Baseline embedding** (`embeddings.py:extract_baseline_embedding`): MFCC mean/std + pitch
  mean/std, L2-normalized. Intentionally simple; swap in a pretrained speaker encoder
  (e.g. ECAPA-TDNN) via `PretrainedSpeakerEncoder` for production accuracy.
- **Verification** (`verification.py`): cosine similarity mapped to `[0, 1]`, compared against a
  **configurable** threshold (`DEFAULT_SIMILARITY_THRESHOLD = 0.55`, override per deployment).
  A given similarity score does not universally mean "same person" — calibrate it.
- **Calibration** (`verification.py:calibrate_threshold`): given labeled genuine/impostor score
  sets, scans candidate thresholds and picks the one that best balances false-accept and
  false-reject rates (an EER-style calibration).

## Prosody analysis (`ml/prosody/`)

- **Features** (`features.py`): pitch/F0 (via `librosa.pyin`), pitch variance, a speech-rate
  proxy (zero-crossing rate), pause frequency (energy-based), RMS energy mean/std, jitter,
  shimmer, and spectral centroid.
- **Anomaly scoring** (`analyzer.py`): compares extracted features against configurable expected
  ranges for natural conversational speech (`EXPECTED_RANGES`) and produces a 0-1 anomaly score.
  This is explicitly **supporting evidence**, combined with other signals in the risk engine —
  never treated as proof of synthetic speech on its own.

## Model management

`GET /api/v1/models` reports the active component/name/version and whether each is a baseline.
Model version strings are configured via environment variables (`DEEPFAKE_MODEL_VERSION`, etc.)
so deployments can track exactly which model produced a given historical risk event.

## Fallback mode

If pretrained models are unavailable for any reason, the complete application still runs end to
end using the classical baselines described above — this is what makes VOICEGUARD runnable on a
normal development machine without GPU access. The dashboard and API responses label baseline
outputs accordingly (`is_baseline: true`, `-untrained` suffixes where relevant).

## Known limitations

- Baseline models are intentionally simple/interpretable and have **not** been benchmarked
  against a standard anti-spoofing dataset in this repository — no accuracy claims are made
  until you train and evaluate against your own data via `train.py` / `evaluate.py`.
- Cross-lingual / Indian-accent robustness depends entirely on the diversity of your training
  data; the architecture is language-agnostic (raw waveform → spectral features) but no claims
  are made about baseline accuracy on any specific accent or language without evaluation data.

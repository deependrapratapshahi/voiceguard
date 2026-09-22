# ml/

Model code for VOICEGUARD's three analysis components. See
[`../docs/ml_pipeline.md`](../docs/ml_pipeline.md) for the full design write-up.

- `deepfake/` — synthetic/cloned speech detector (baseline classical model + pluggable
  pretrained-encoder interface). `train.py` / `evaluate.py` / `inference.py` / `model.py` /
  `dataset.py` / `preprocessing.py`.
- `speaker/` — speaker embedding extraction + verification (`embeddings.py`,
  `verification.py`, `inference.py`).
- `prosody/` — prosodic/acoustic feature extraction and anomaly scoring (`features.py`,
  `analyzer.py`).
- `common/` — shared feature-extraction helpers (`audio.py`) used by the baseline models.

All modules are importable both from the FastAPI backend (`app/services/*_service.py`) and
standalone via the CLI training/evaluation scripts.

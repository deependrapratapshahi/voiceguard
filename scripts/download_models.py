"""
Guidance script for setting up pretrained speech encoders.

This script does NOT automatically download any model weights or
datasets on your behalf -- some pretrained speech models and most
anti-spoofing datasets (e.g. ASVspoof) require accepting a license or
data-use agreement first. Instead, this script prints exactly where to
get them and where to place the resulting files.

Run: python scripts/download_models.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "ml_artifacts"

GUIDANCE = f"""
VOICEGUARD — Pretrained Model Setup Guide
==========================================

VOICEGUARD runs in BASELINE MODE by default (classical ML over log-mel /
MFCC features, no downloads required). To optionally enable stronger
pretrained speech encoders:

1. Synthetic-voice detection encoder (optional):
   - e.g. a wav2vec2 / HuBERT / WavLM checkpoint from Hugging Face Hub
     (https://huggingface.co/models) -- pick a checkpoint whose license
     permits your use case.
   - After downloading, place the encoder + a trained classification
     head at:
       {MODEL_DIR / 'deepfake_pretrained_head.pt'}
   - Set USE_PRETRAINED_ENCODERS=true in your .env file.

2. Speaker verification encoder (optional):
   - e.g. an ECAPA-TDNN checkpoint (SpeechBrain) or another pretrained
     speaker-embedding model whose license permits your use case.
   - Wire it into ml/speaker/embeddings.py's PretrainedSpeakerEncoder.

3. Anti-spoofing training datasets (for ml/deepfake/train.py):
   - ASVspoof (https://www.asvspoof.org) and similar datasets require
     registering and accepting their terms of use before download.
   - After obtaining a dataset, place files as:
       datasets/real/<file>.wav       (bona fide speech)
       datasets/synthetic/<file>.wav  (spoofed/synthetic speech)
   - Then run: python scripts/validate_dataset.py

Until pretrained weights are wired in, the system automatically uses
the classical baseline models and clearly labels itself as
"DEMO / BASELINE MODEL" in API responses and the dashboard.
"""

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(GUIDANCE)

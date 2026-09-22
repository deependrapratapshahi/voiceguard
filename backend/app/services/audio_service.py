"""
Audio preprocessing service.

Pipeline: decode -> resample to 16kHz mono -> normalize -> VAD/silence
removal -> chunk into overlapping windows for streaming-style inference.

This module never assumes the entire call is available up front -- the
`chunk_stream` generator can be fed incrementally from the WebSocket
handler.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Generator, List

import numpy as np
import librosa
import soundfile as sf

from app.config import get_settings

settings = get_settings()


@dataclass
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    is_voiced: bool


def decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary supported audio bytes into a float32 waveform + sr."""
    with io.BytesIO(raw_bytes) as buf:
        samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)  # downmix to mono
    return samples, sr


def resample_and_normalize(samples: np.ndarray, sr: int) -> np.ndarray:
    target_sr = settings.TARGET_SAMPLE_RATE
    if sr != target_sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1e-8:
        samples = samples / peak
    return samples.astype(np.float32)


def remove_silence(samples: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    if samples.size == 0:
        return samples
    intervals = librosa.effects.split(samples, top_db=top_db)
    if len(intervals) == 0:
        return samples
    voiced = np.concatenate([samples[s:e] for s, e in intervals])
    return voiced if voiced.size else samples


def simple_vad_mask(samples: np.ndarray, frame_length: int = 1024, hop_length: int = 256,
                     energy_threshold: float = 0.01) -> np.ndarray:
    """
    Lightweight energy-based VAD baseline. Returns a boolean mask at the
    frame level. Designed to be swappable for a learned VAD model later.
    """
    if samples.size < frame_length:
        return np.array([samples.size > 0 and np.sqrt(np.mean(samples ** 2)) > energy_threshold])
    rms = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
    return rms > energy_threshold


def chunk_waveform(samples: np.ndarray, sr: int) -> List[AudioChunk]:
    """Split a full waveform into overlapping chunks for analysis."""
    chunk_len = int(settings.CHUNK_DURATION_SECONDS * sr)
    hop_len = int((settings.CHUNK_DURATION_SECONDS - settings.CHUNK_OVERLAP_SECONDS) * sr)
    hop_len = max(hop_len, 1)

    chunks: List[AudioChunk] = []
    pos = 0
    while pos < len(samples):
        window = samples[pos: pos + chunk_len]
        if len(window) < int(0.5 * sr):  # discard trailing fragment shorter than 0.5s
            break
        voiced_mask = simple_vad_mask(window)
        is_voiced = bool(np.mean(voiced_mask) > 0.15) if len(voiced_mask) else False
        chunks.append(
            AudioChunk(
                samples=window,
                sample_rate=sr,
                start_time=pos / sr,
                end_time=(pos + len(window)) / sr,
                is_voiced=is_voiced,
            )
        )
        pos += hop_len
    return chunks


def preprocess_full_audio(raw_bytes: bytes) -> List[AudioChunk]:
    """End-to-end pipeline used by the REST upload-analysis endpoint."""
    samples, sr = decode_audio_bytes(raw_bytes)
    samples = resample_and_normalize(samples, sr)
    samples = remove_silence(samples, top_db=30.0)
    return chunk_waveform(samples, settings.TARGET_SAMPLE_RATE)


class StreamingPreprocessor:
    """
    Stateful helper for the WebSocket pipeline: buffers incoming raw PCM
    chunks and yields fixed-size overlapping windows as soon as enough
    audio has accumulated, without waiting for the call to end.
    """

    def __init__(self):
        self._buffer = np.zeros(0, dtype=np.float32)
        self._sr = settings.TARGET_SAMPLE_RATE
        self._chunk_len = int(settings.CHUNK_DURATION_SECONDS * self._sr)
        self._hop_len = max(
            int((settings.CHUNK_DURATION_SECONDS - settings.CHUNK_OVERLAP_SECONDS) * self._sr), 1
        )
        self._elapsed = 0.0

    def push(self, raw_bytes: bytes) -> Generator[AudioChunk, None, None]:
        samples, sr = decode_audio_bytes(raw_bytes)
        samples = resample_and_normalize(samples, sr)
        self._buffer = np.concatenate([self._buffer, samples])

        while len(self._buffer) >= self._chunk_len:
            window = self._buffer[: self._chunk_len]
            voiced_mask = simple_vad_mask(window)
            is_voiced = bool(np.mean(voiced_mask) > 0.15) if len(voiced_mask) else False
            yield AudioChunk(
                samples=window,
                sample_rate=self._sr,
                start_time=self._elapsed,
                end_time=self._elapsed + self._chunk_len / self._sr,
                is_voiced=is_voiced,
            )
            self._buffer = self._buffer[self._hop_len:]
            self._elapsed += self._hop_len / self._sr

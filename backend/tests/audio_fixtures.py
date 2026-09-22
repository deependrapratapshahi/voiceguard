"""
Shared helper for generating small synthetic (sine-wave) WAV audio for
tests. No real voice/speech data is used anywhere in this test suite --
these are simple tones used only to exercise the audio pipeline's
decode/preprocess/inference code paths.
"""
import io
import struct
import wave


def make_wav_bytes(duration_seconds: float = 2.0, sr: int = 16000, freq: float = 220.0) -> bytes:
    """Generates a small synthetic sine-wave WAV in memory."""
    n_samples = int(duration_seconds * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            import math

            value = int(6000 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", value)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def make_corrupt_audio_bytes() -> bytes:
    """Bytes that are not a valid audio file of any supported format."""
    return b"this-is-not-a-real-audio-file-just-garbage-bytes-1234567890"

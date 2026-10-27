"""ffmpeg wrappers and WAV slicing. Everything downstream works on 16 kHz mono PCM16."""
from __future__ import annotations

import io
import json
import subprocess
import wave
from pathlib import Path

SAMPLE_RATE = 16000


def to_wav16k(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )
    return dst


def duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono PCM16"
        return w.readframes(w.getnframes())


def slice_wav(pcm: bytes, start: float, end: float) -> bytes:
    """Return a complete WAV file (bytes) for pcm[start:end] seconds."""
    a, b = int(start * SAMPLE_RATE) * 2, int(end * SAMPLE_RATE) * 2
    buf = io.BytesIO()
    write_wav(buf, pcm[a:b])
    return buf.getvalue()


def write_wav(dst, pcm: bytes) -> None:
    """Write 16 kHz mono PCM16 bytes as a WAV file to a path or a binary file object."""
    with wave.open(dst if hasattr(dst, "write") else str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)

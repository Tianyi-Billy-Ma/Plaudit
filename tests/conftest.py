import math
import struct
import wave
from pathlib import Path

import pytest

from plaudit.models import Segment


@pytest.fixture
def clip(tmp_path) -> Path:
    """4 s of two alternating tones at 8 kHz stereo, so the pipeline has to resample and downmix."""
    path = tmp_path / "meeting.wav"
    rate = 8000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(rate * 4):
            f = 440 if (i // rate) % 2 == 0 else 660
            v = int(8000 * math.sin(2 * math.pi * f * i / rate))
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))
    return path


class FakeDiarizer:
    def diarize(self, wav):
        return [Segment(0.0, 2.0, "Speaker 1"), Segment(2.0, 4.0, "Speaker 2")]


class FakeTranscriber:
    def transcribe(self, wav, turns):
        for i, t in enumerate(turns):
            t.text = f"turn {i} 中文 and English"
        return turns


class FakeSummarizer:
    def summarize(self, text, glossary=(), cache_file=None):
        return "## 摘要\n- two turns\n## 决定\n无\n## 行动项\n- nobody: nothing"


@pytest.fixture
def cfg(tmp_path):
    from plaudit import config

    (tmp_path / "plaudit.yaml").write_text(
        f"data_dir: {tmp_path}/data\nglossary: {tmp_path}/glossary.txt\n"
        f"sources:\n  folder:\n    path: {tmp_path}/inbox\n"
        f"outputs:\n  markdown:\n    dir: {tmp_path}/notes\n  notion:\n    database_id: null\n"
        f"min_free_mem_gb: 0\n"
    )
    return config.load(tmp_path / "plaudit.yaml")

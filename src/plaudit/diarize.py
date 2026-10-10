"""Speaker diarization backends. Each returns speaker turns as Segments with empty text."""
from __future__ import annotations

import logging
from pathlib import Path

from .audio import duration_seconds
from .models import Segment

log = logging.getLogger(__name__)


class NoDiarizer:
    """Whole recording is one speaker turn; chunked so ASR never sees an hour-long tensor."""

    def __init__(self, cfg: dict):
        self.max_turn = cfg["diarization"]["max_turn_seconds"]

    def diarize(self, wav: Path) -> list[Segment]:
        total = duration_seconds(wav)
        return split_turns([Segment(0.0, total, "Speaker 1")], self.max_turn)


class NemotronDiarizer:
    """nvidia/Nemotron-3-Diarization through transformers (frame logits every 10 ms, up to 8 speakers)."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        self.torch = torch
        self.max_turn = cfg["diarization"]["max_turn_seconds"]
        model_id = cfg["diarization"]["model"]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto").eval()

    def diarize(self, wav: Path) -> list[Segment]:
        import numpy as np

        from .audio import SAMPLE_RATE, read_pcm

        audio = np.frombuffer(read_pcm(wav), dtype=np.int16).astype(np.float32) / 32768.0
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE).to(self.model.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        raw = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        turns = [Segment(float(s["Start"]), float(s["End"]), f"Speaker {int(s['Speaker']) + 1}") for s in raw]
        return split_turns(merge_turns(sorted(turns, key=lambda s: s.start)), self.max_turn)


def merge_turns(turns: list[Segment], gap: float = 0.5) -> list[Segment]:
    """Join consecutive segments of the same speaker separated by less than `gap` seconds."""
    out: list[Segment] = []
    for t in turns:
        if out and out[-1].speaker == t.speaker and t.start - out[-1].end <= gap:
            out[-1].end = max(out[-1].end, t.end)
        else:
            out.append(Segment(t.start, t.end, t.speaker))
    return out


def split_turns(turns: list[Segment], max_seconds: float) -> list[Segment]:
    out: list[Segment] = []
    for t in turns:
        s = t.start
        while t.end - s > max_seconds:
            out.append(Segment(s, s + max_seconds, t.speaker))
            s += max_seconds
        if t.end - s > 0.2:  # drop sub-200 ms slivers
            out.append(Segment(s, t.end, t.speaker))
    return out


BACKENDS = {"none": NoDiarizer, "nemotron": NemotronDiarizer}


def build(cfg: dict):
    return BACKENDS[cfg["diarization"]["backend"]](cfg)

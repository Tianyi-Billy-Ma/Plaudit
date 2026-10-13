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
        from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
            Nemotron3DiarizationSpeakerCache,
        )

        from .audio import SAMPLE_RATE, read_pcm

        audio = np.frombuffer(read_pcm(wav), dtype=np.int16).astype(np.float32) / 32768.0
        # Offline mode embeds the whole recording at once (OOM past ~1 h). Stream it instead, in chunks of the
        # offline size with the offline cache sizes, so the result matches offline while memory stays flat.
        cfg, proc, fe = self.model.config, self.processor, self.processor.feature_extractor
        proc.streaming_modes["offline"] = (cfg.chunk_length, cfg.chunk_right_context)
        proc.set_streaming_mode("offline")
        cache = Nemotron3DiarizationSpeakerCache(cfg.streaming_config, fifo_length=cfg.fifo_length,
                                                 speaker_cache_update_period=cfg.speaker_cache_update_period)
        logits = []
        with self.torch.inference_mode():
            for i, (start, stop, last) in enumerate(stream_chunks(
                    len(audio), proc.num_samples_first_audio_chunk, proc.num_samples_per_audio_chunk,
                    proc.num_mel_frames_per_step * fe.hop_length, fe.n_fft // 2)):
                inputs = proc(audio[start:stop], sampling_rate=SAMPLE_RATE, is_streaming=True,
                              is_first_audio_chunk=i == 0, is_last_audio_chunk=last)
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)
                out = self.model(**inputs, speaker_cache=cache)
                cache = out.speaker_cache
                logits.append(out.logits)  # chunk frames only; the look-ahead ones come back in the next chunk
        raw = self.processor.extract_speaker_dict(self.torch.cat(logits, 1))[0]
        turns = [Segment(float(s["Start"]), float(s["End"]), f"Speaker {int(s['Speaker']) + 1}") for s in raw]
        return split_turns(merge_turns(sorted(turns, key=lambda s: s.start)), self.max_turn)


def stream_chunks(total: int, first: int, later: int, step: int, offset: int) -> list[tuple[int, int, bool]]:
    """Sample ranges (start, stop, is_last) of a Nemotron streaming session over `total` samples: the first chunk
    holds `first` samples from 0, every later one `later` samples starting `offset` before its frame cursor, which
    advances `step` samples per chunk. The last chunk takes whatever remains."""
    out, start, n = [], 0, first
    while True:
        if start + n >= total:
            return out + [(start, total, True)]
        out.append((start, start + n, False))
        start, n = step * len(out) - offset, later

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

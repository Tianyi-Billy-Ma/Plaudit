"""Transcription backends. Each fills in Segment.text for a list of speaker turns."""
from __future__ import annotations

import logging
from pathlib import Path

from .audio import read_pcm, slice_wav
from .config import glossary, secret
from .http import post_multipart
from .models import Segment

log = logging.getLogger(__name__)


class QwenASRTranscriber:
    """Qwen/Qwen3-ASR-1.7B via the qwen_asr package, batched over speaker turns."""

    def __init__(self, cfg: dict):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.cfg = cfg
        self.model = Qwen3ASRModel.from_pretrained(
            cfg["transcription"]["model"], dtype=torch.bfloat16, device_map="cuda:0",
            max_inference_batch_size=8, max_new_tokens=512,
        )

    def transcribe(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        import numpy as np

        from .audio import SAMPLE_RATE

        pcm = read_pcm(wav)
        clips = []
        for t in turns:
            a, b = int(t.start * SAMPLE_RATE) * 2, int(t.end * SAMPLE_RATE) * 2
            clips.append((np.frombuffer(pcm[a:b], dtype=np.int16).astype(np.float32) / 32768.0, SAMPLE_RATE))
        # qwen_asr>=0.0.6: `context` is the system prompt of each request, the documented way to bias terms
        results = self.model.transcribe(audio=clips, context=", ".join(glossary(self.cfg)), language=self.cfg["language"])
        for t, r in zip(turns, results):
            t.text = r.text.strip()
        return turns


class OpenAICompatTranscriber:
    """Any /v1/audio/transcriptions endpoint (whisper.cpp server, vLLM, OpenAI...). Glossary goes in `prompt`."""

    def __init__(self, cfg: dict):
        tc = cfg["transcription"]
        self.url = tc["base_url"].rstrip("/") + "/audio/transcriptions"
        self.model = tc["model"]
        self.headers = {}
        if key := secret(tc.get("api_key_env")):
            self.headers["Authorization"] = f"Bearer {key}"
        self.prompt = ", ".join(glossary(cfg))

    def transcribe(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        pcm = read_pcm(wav)
        for t in turns:
            fields = {"model": self.model, "response_format": "json"}
            if self.prompt:
                fields["prompt"] = self.prompt
            resp = post_multipart(self.url, fields, "file", "turn.wav", slice_wav(pcm, t.start, t.end), self.headers)
            t.text = resp.get("text", "").strip()
        return turns


BACKENDS = {"qwen_asr": QwenASRTranscriber, "openai": OpenAICompatTranscriber}


def build(cfg: dict):
    return BACKENDS[cfg["transcription"]["backend"]](cfg)

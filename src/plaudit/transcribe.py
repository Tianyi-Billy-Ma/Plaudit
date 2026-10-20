"""Transcription backends. Each fills in Segment.text for a list of speaker turns."""
from __future__ import annotations

import logging
from pathlib import Path

from . import device
from .audio import read_pcm, slice_wav
from .config import glossary, secret
from .http import post_multipart
from .models import Segment

log = logging.getLogger(__name__)
LANGUAGE_CODES = {"Chinese": "zh", "English": "en"}  # config uses Qwen's names; HTTP endpoints want ISO codes


class QwenASRTranscriber:
    """Qwen/Qwen3-ASR-1.7B-hf through transformers (>=5.16 ships the model), batched over speaker turns."""

    BATCH = 8

    def __init__(self, cfg: dict):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.cfg = cfg
        model_id = cfg["transcription"]["model"]
        dev, dtype = device.pick()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()

    def transcribe(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        import numpy as np
        import torch

        from .audio import SAMPLE_RATE

        pcm = read_pcm(wav)
        clips = []
        for t in turns:
            a, b = int(t.start * SAMPLE_RATE) * 2, int(t.end * SAMPLE_RATE) * 2
            clips.append(np.frombuffer(pcm[a:b], dtype=np.int16).astype(np.float32) / 32768.0)
        # the glossary goes in as the system prompt (`prompt`), the documented way to bias terms
        prompt = ", ".join(glossary(self.cfg)) or None
        for i in range(0, len(clips), self.BATCH):
            batch = clips[i:i + self.BATCH]
            inputs = self.processor.apply_transcription_request(
                batch, prompt=prompt, language=self.cfg["language"],
            ).to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=512)
            texts = self.processor.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")
            for t, text in zip(turns[i:i + self.BATCH], texts):
                t.text = text.strip()
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
        # Whisper-style endpoints take ISO codes; without one they guess per request and may switch script or translate
        self.language = LANGUAGE_CODES.get(cfg["language"], cfg["language"])

    def transcribe(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        pcm = read_pcm(wav)
        for t in turns:
            fields = {"model": self.model, "response_format": "json"}
            if self.prompt:
                fields["prompt"] = self.prompt
            if self.language:
                fields["language"] = self.language
            resp = post_multipart(self.url, fields, "file", "turn.wav", slice_wav(pcm, t.start, t.end), self.headers)
            t.text = resp.get("text", "").strip()
        return turns


BACKENDS = {"qwen_asr": QwenASRTranscriber, "openai": OpenAICompatTranscriber}


def build(cfg: dict):
    return BACKENDS[cfg["transcription"]["backend"]](cfg)

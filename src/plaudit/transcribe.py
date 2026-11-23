"""Transcription backends. Each fills in Segment.text for a list of speaker turns."""
from __future__ import annotations

import logging
import math
import tempfile
import time
import urllib.error
from pathlib import Path

from . import device
from .audio import duration_seconds, read_pcm, slice_wav, to_m4a
from .config import glossary, secret
from .http import HttpError, get_json, post_multipart
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
        import torch

        pcm = read_pcm(wav)
        clips = [clip_float(pcm, t) for t in turns]
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


class WhisperTranscriber:
    """Whisper through faster-whisper (CTranslate2, CPU or CUDA). `model` is a size (`large-v3`, `turbo`) or a path."""

    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel

        # faster-whisper's own device pick (CUDA else CPU) equals device.pick() here, since CTranslate2 has no MPS,
        # and it needs no torch: the whisper extra must stay usable on a CPU box without it.
        self.model = WhisperModel(cfg["transcription"]["model"])
        self.language = LANGUAGE_CODES.get(cfg["language"], cfg["language"])
        self.prompt = ", ".join(glossary(cfg)) or None

    def transcribe(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        pcm = read_pcm(wav)
        for t in turns:
            segments, _ = self.model.transcribe(clip_float(pcm, t), language=self.language, initial_prompt=self.prompt)
            t.text = " ".join(s.text.strip() for s in segments)
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


class DiarizedAPITranscriber(OpenAICompatTranscriber):
    """Diarization and transcription in one /v1/audio/transcriptions call with response_format=diarized_json (vLLM
    with MOSS-Transcribe-Diarize, OpenAI gpt-4o-transcribe-diarize...). The whole recording goes up as one m4a, split
    only past `max_segment_minutes`. With `async` it asks for `Prefer: respond-async` and polls GET .../{id}; a server
    that ignores the header and answers at once is used as is. The pipeline uses this object as its diarizer too."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        tc = cfg["transcription"]
        self.max_seconds = tc["max_segment_minutes"] * 60
        self.use_async, self.poll = tc["async"], tc["poll_seconds"]
        self.timeout = tc["timeout_minutes"] * 60

    def diarize(self, wav: Path) -> list[Segment]:
        total = duration_seconds(wav)
        n = max(1, math.ceil(total / self.max_seconds))  # parts of equal length
        out: list[Segment] = []
        with tempfile.TemporaryDirectory() as d:
            for i in range(n):
                start = i * total / n
                part = to_m4a(wav, Path(d) / f"part{i}.m4a", start, total / n if n > 1 else None)
                log.info("uploading part %d/%d (%.1f MB)", i + 1, n, part.stat().st_size / 1e6)
                # shortcut: speakers are not matched across parts (labels get " (part N)"); name them with voices/
                out += parse_diarized(self._request(part), start, f" (part {i + 1})" if i else "")
        return out

    def transcribe(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        return turns  # diarize() already filled in the text

    def _request(self, part: Path) -> dict:
        fields = {"model": self.model, "response_format": "diarized_json"}
        if self.prompt:
            fields["prompt"] = self.prompt
        if self.language:
            fields["language"] = self.language
        headers = {**self.headers, **({"Prefer": "respond-async"} if self.use_async else {})}
        try:
            resp = post_multipart(self.url, fields, "file", "audio.m4a", part.read_bytes(), headers,
                                  timeout=self.timeout, content_type="audio/mp4")
        except HttpError as e:
            if e.status in (400, 415, 422) and "diarized" in str(e).lower():
                raise RuntimeError(f"{self.url} does not support response_format=diarized_json (model {self.model}): "
                                   f"{e}. Use a diarizing model, or diarization.backend: nemotron with "
                                   f"transcription.backend: openai") from None
            raise
        if "segments" in resp or "text" in resp or "id" not in resp:
            return resp  # a synchronous answer, also when the server ignored Prefer: respond-async
        return self._wait(resp["id"])

    def _wait(self, job_id: str) -> dict:
        deadline, errors = time.time() + self.timeout, 0
        while time.time() < deadline:
            time.sleep(self.poll)
            try:
                job = get_json(f"{self.url}/{job_id}", self.headers)
                errors = 0
            except (HttpError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
                errors += 1  # a proxy hiccup should not lose a job that is still running server side
                if (isinstance(e, HttpError) and e.status < 500 and e.status != 429) or errors >= 5:
                    raise
                continue
            status = job.get("status")
            if status in ("failed", "cancelled", "canceled", "error"):
                raise RuntimeError(f"transcription job {job_id} {status}: {job.get('error')}")
            if status in ("completed", "succeeded", "done") or "segments" in job:
                return job.get("result") or job.get("response") or job
        raise TimeoutError(f"transcription job {job_id} not done after {self.timeout / 60:.0f} min")


def parse_diarized(resp: dict, offset: float = 0.0, suffix: str = "") -> list[Segment]:
    """diarized_json segments -> Segments with Plaudit's `Speaker N` labels, numbered by first appearance."""
    labels: dict = {}
    out = []
    for s in resp.get("segments") or []:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = labels.setdefault(s.get("speaker"), f"Speaker {len(labels) + 1}{suffix}")
        out.append(Segment(offset + float(s["start"]), offset + float(s["end"]), spk, text))
    if not out and (resp.get("text") or "").strip():  # no segments at all: keep the text rather than lose it
        out.append(Segment(offset, offset, f"Speaker 1{suffix}", resp["text"].strip()))
    return out


def clip_float(pcm: bytes, t: Segment):
    """One turn of 16 kHz mono int16 PCM as float32 in [-1, 1]."""
    import numpy as np

    from .audio import SAMPLE_RATE

    a, b = int(t.start * SAMPLE_RATE) * 2, int(t.end * SAMPLE_RATE) * 2
    return np.frombuffer(pcm[a:b], dtype=np.int16).astype(np.float32) / 32768.0


BACKENDS = {"qwen_asr": QwenASRTranscriber, "whisper": WhisperTranscriber, "openai": OpenAICompatTranscriber,
            "openai_diarized": DiarizedAPITranscriber}


def build(cfg: dict):
    return BACKENDS[cfg["transcription"]["backend"]](cfg)

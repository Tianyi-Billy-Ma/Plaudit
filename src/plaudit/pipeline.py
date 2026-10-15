"""One recording in, transcript + summary out. No queue, no retries: that is daemon.py's job."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import audio, config, diarize, outputs, transcribe
from .models import Segment, Transcript
from .summarize import Summarizer

log = logging.getLogger(__name__)


class Pipeline:
    """Loads the models once; process() can then be called for many files."""

    def __init__(self, cfg: dict, diarizer=None, transcriber=None, summarizer=None, sinks=None):
        self.cfg = cfg
        self.diarizer = diarizer or diarize.build(cfg)
        self.transcriber = transcriber or transcribe.build(cfg)
        self.summarizer = summarizer or (Summarizer(cfg) if cfg["summary"]["base_url"] else None)
        self.sinks = outputs.build(cfg) if sinks is None else sinks

    def process(self, src: Path, meta: dict, workdir: Path, audio_link: str | None = None) -> dict:
        workdir.mkdir(parents=True, exist_ok=True)
        wav = audio.to_wav16k(src, workdir / "audio16k.wav")
        meta = {**meta, "duration_min": round(audio.duration_seconds(wav) / 60, 1)}
        saved = workdir / "transcript.json"
        if saved.exists():  # a retry after the summary or an output failed: 20+ min of diarization and ASR are done
            log.info("%s: reusing %s", meta["title"], saved)
            d = json.loads(saved.read_text())
            transcript = Transcript([Segment(**x) for x in d["segments"]], d["language"])
        else:
            turns = self.diarizer.diarize(wav)
            log.info("%s: %d speaker turns", meta["title"], len(turns))
            transcript = Transcript(self.transcriber.transcribe(wav, turns), self.cfg["language"])
            saved.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1))
        summary = (self.summarizer.summarize(transcript.as_text(), config.glossary(self.cfg), workdir / "summary-cache.json")
                   if self.summarizer else "")
        (workdir / "notes.md").write_text(outputs.render_markdown(meta, summary, transcript, audio_link))
        links = [sink.publish(meta, summary, transcript, audio_link) for sink in self.sinks]
        wav.unlink()
        return {"meta": meta, "summary": summary, "transcript": transcript, "links": links}

"""Phase 1, tests 3 and 4: run diarization + ASR on one clip, with and without the glossary, and report
time, peak GPU memory and both transcripts. Usage (inside the pipeline container):
    python scripts/phase1_bench.py /plaudit/test.m4a [--diarization nemotron|none]
Writes transcript_with_glossary.md and transcript_no_glossary.md next to the clip.
"""
import argparse
import logging
import tempfile
import time
from pathlib import Path

import torch

from plaudit import audio, config, diarize, transcribe
from plaudit.models import Transcript

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("--diarization", default="nemotron")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO)

cfg = config.load()
cfg["diarization"]["backend"] = args.diarization
clip = Path(args.clip)
wav = audio.to_wav16k(clip, Path(tempfile.mkdtemp()) / "a.wav")
print(f"clip: {audio.duration_seconds(wav) / 60:.1f} min")

torch.cuda.reset_peak_memory_stats()
t = time.time()
d = diarize.build(cfg)
turns = d.diarize(wav)
print(f"diarization: {len(turns)} turns, {len({x.speaker for x in turns})} speakers, {time.time() - t:.0f}s, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
del d
torch.cuda.empty_cache()

asr = transcribe.build(cfg)
for label, gl in (("with_glossary", cfg["glossary"]), ("no_glossary", "/nonexistent")):
    cfg["glossary"] = gl
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    segs = asr.transcribe(wav, [type(x)(x.start, x.end, x.speaker) for x in turns])
    print(f"asr {label}: {time.time() - t:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    out = clip.with_name(f"transcript_{label}.md")
    out.write_text(Transcript(segs).as_text())
    print(f"  -> {out}")

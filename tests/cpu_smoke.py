"""Manual check that the local models run without a GPU: `python tests/cpu_smoke.py two_speakers.wav`
(16 kHz mono). Not collected by pytest: it downloads the two models and takes a minute on a laptop."""
import sys
import time
from pathlib import Path

from plaudit import config, diarize, transcribe

cfg = config.load("/nonexistent/plaudit.yaml")
cfg["diarization"]["backend"] = "nemotron"
wav = Path(sys.argv[1])
t0 = time.time()
turns = diarize.build(cfg).diarize(wav)
print(f"diarize {time.time() - t0:.1f}s, {len(turns)} turns, speakers {sorted({t.speaker for t in turns})}")
t0 = time.time()
for s in transcribe.build(cfg).transcribe(wav, turns):
    print(f"  {s.start:6.1f} {s.speaker}: {s.text}")
print(f"asr {time.time() - t0:.1f}s")
assert len({t.speaker for t in turns}) >= 2 and all(t.text for t in turns)

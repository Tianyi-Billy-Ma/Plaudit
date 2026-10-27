"""Who is Speaker 2? One audio sample per person in a folder, one embedding model, cosine similarity.
Each backend turns a 16 kHz float32 clip into one unit vector; the matching, the renaming and the enrolment
from a finished job are shared."""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from . import device
from .audio import SAMPLE_RATE, read_pcm, to_wav16k, write_wav
from .models import Segment
from .transcribe import clip_float

log = logging.getLogger(__name__)
MAX_SECONDS = 30  # of audio per speaker and per sample; the models are trained on short segments
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".webm", ".wma"}


class ReDimNet2Embedder:
    """ReDimNet2 (Interspeech 2026, MIT) through torch.hub, pinned to one commit; weights from the project's
    GitHub release into ~/.cache/torch. `model` is the size b0..b6; b3 and b6 have the VoxBlink2+VoxCeleb2+CN-Celeb2
    weights, the other sizes exist for VoxCeleb2 only."""

    REPO = "PalabraAI/redimnet2:c5bbe0b76e37df698c403f8844e41304ceab6307"

    def __init__(self, cfg: dict):
        import torch

        self.torch = torch
        size = cfg["speakers"]["model"]
        self.dev = device.pick()[0]
        self.model = torch.hub.load(self.REPO, "redimnet2", model_name=size, train_type="lm", trust_repo=True,
                                    dataset="vb2+vox2+cnc2_v0" if size in ("b3", "b6") else "vox2").to(self.dev).eval()

    def embed(self, clip):
        with self.torch.inference_mode():
            return unit(self.model(self.torch.from_numpy(clip)[None].to(self.dev))[0].float().cpu().numpy())


class XVectorEmbedder:
    """Any `*ForXVector` model on Hugging Face through transformers (microsoft/wavlm-base-plus-sv,
    microsoft/unispeech-sat-large-sv ...)."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.torch = torch
        model_id = cfg["speakers"]["model"]
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioXVector.from_pretrained(model_id).to(device.pick()[0]).eval()

    def embed(self, clip):
        inputs = self.extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            return unit(self.model(**inputs).embeddings[0].float().cpu().numpy())


class SherpaEmbedder:
    """sherpa-onnx (`pip install plaudit[sherpa]`, CPU): `model` is the path of an ONNX speaker model, e.g. one of
    the 3D-Speaker (ERes2NetV2, CAM++), WeSpeaker or NeMo TitaNet exports at
    https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models"""

    def __init__(self, cfg: dict):
        import sherpa_onnx

        path = Path(cfg["speakers"]["model"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"speakers.model: {path} is not an ONNX file")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=2))

    def embed(self, clip):
        import numpy as np

        stream = self.extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, clip)
        stream.input_finished()
        return unit(np.asarray(self.extractor.compute(stream), dtype=np.float32))


# backend -> (class, default cosine threshold). The thresholds come from each model's own examples or our
# measurement; `speakers.threshold` overrides. Scores are logged per job so a wrong one is visible.
BACKENDS = {"redimnet2": (ReDimNet2Embedder, 0.5), "xvector": (XVectorEmbedder, 0.86), "sherpa": (SherpaEmbedder, 0.5)}


def unit(v):
    import numpy as np

    return v / (np.linalg.norm(v) or 1.0)


def samples(folder: Path) -> dict[str, list[Path]]:
    """`张三.wav` and `张三.2.m4a` are two samples of 张三. Hidden files are ignored."""
    out: dict[str, list[Path]] = {}
    for p in sorted(folder.iterdir()) if folder.is_dir() else ():
        if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("."):
            out.setdefault(p.stem.split(".")[0], []).append(p)
    return out


class Namer:
    """Renames diarized speakers to the people in the voices folder. The folder is read fresh for every job,
    the model is loaded on the first job that has a sample to compare with."""

    def __init__(self, cfg: dict, embedder=None):
        self.cfg = cfg["speakers"]
        self.embedder = embedder
        self.threshold = self.cfg["threshold"] or BACKENDS[self.cfg["backend"]][1]

    def name(self, wav: Path, turns: list[Segment]) -> list[Segment]:
        library = samples(Path(self.cfg["voices"]))
        if not library:
            return turns
        self.embedder = self.embedder or BACKENDS[self.cfg["backend"]][0]({"speakers": self.cfg})
        people = {name: unit(sum(self._embed_file(p) for p in paths)) for name, paths in library.items()}
        pcm = read_pcm(wav)
        voices = {}
        for t in turns:  # up to MAX_SECONDS of each speaker's own turns
            voices.setdefault(t.speaker, []).append(t)
        scores = []
        for label, own in voices.items():
            vec = self.embedder.embed(concat(pcm, own))
            scores += [(float(vec @ v), label, name) for name, v in people.items()]
        names, taken = {}, set()
        for score, label, name in sorted(scores, reverse=True):  # best pairs first, one name per speaker
            if score >= self.threshold and label not in names and name not in taken:
                names[label], taken = name, taken | {name}
        for label in voices:
            best = max((s, n) for s, l, n in scores if l == label)
            log.info("%s -> %s (best match %s at %.2f, threshold %.2f)", label, names.get(label, "unknown"), best[1], best[0], self.threshold)
        for t in turns:
            t.speaker = names.get(t.speaker, t.speaker)
        return turns

    def _embed_file(self, path: Path):
        with tempfile.TemporaryDirectory() as d:  # shortcut: samples are re-embedded every job, cache the vectors if a folder grows past dozens
            pcm = read_pcm(to_wav16k(path, Path(d) / "sample.wav"))
        return self.embedder.embed(clip_float(pcm, Segment(0.0, min(MAX_SECONDS, len(pcm) / 2 / SAMPLE_RATE), "")))


def concat(pcm: bytes, turns: list[Segment]):
    """Float clip of the given turns joined, cut at MAX_SECONDS."""
    import numpy as np

    return np.concatenate([clip_float(pcm, t) for t in turns])[:MAX_SECONDS * SAMPLE_RATE]


def sample_path(cfg: dict, name: str) -> Path:
    """Where the next sample of `name` goes: voices/张三.wav, then 张三.2.wav, 张三.3.wav ..."""
    voices = Path(cfg["speakers"]["voices"])
    voices.mkdir(parents=True, exist_ok=True)
    if "/" in name or name.startswith("."):
        raise ValueError(f"{name!r} is not a usable speaker name")
    n = 1 + len(samples(voices).get(name, []))
    return voices / (f"{name}.wav" if n == 1 else f"{name}.{n}.wav")


def add_sample(cfg: dict, name: str, src: Path) -> Path:
    """Convert a recording of `name` into the voices folder; the first MAX_SECONDS are what gets compared."""
    dst = sample_path(cfg, name)
    with tempfile.TemporaryDirectory() as d:
        pcm = read_pcm(to_wav16k(src, Path(d) / "sample.wav"))
    write_wav(dst, pcm[:MAX_SECONDS * SAMPLE_RATE * 2])
    return dst


def rename_job(cfg: dict, workdir: Path, names: dict[str, str], links: list[str], enrol: bool = False) -> None:
    """Replace speaker labels in a finished job's transcript.json, notes.md and Markdown note (not in pages already
    published elsewhere). With `enrol`, save up to MAX_SECONDS of each renamed speaker's turns in the voices folder."""
    import json

    from .models import Transcript

    saved = workdir / "transcript.json"
    d = json.loads(saved.read_text())
    transcript = Transcript([Segment(**x) for x in d["segments"]], d["language"])
    labels = {s.speaker for s in transcript.segments}
    if unknown := set(names) - labels:
        raise ValueError(f"no such speaker in this job: {', '.join(sorted(unknown))}; it has {', '.join(sorted(labels))}")
    if enrol:
        original = next(p for p in workdir.iterdir() if p.suffix.lower() in AUDIO_EXT and p.name != "audio16k.wav")
        with tempfile.TemporaryDirectory() as d:
            pcm = read_pcm(to_wav16k(original, Path(d) / "audio16k.wav"))
        for label, name in names.items():
            dst = sample_path(cfg, name)
            clip = concat(pcm, [s for s in transcript.segments if s.speaker == label])
            write_wav(dst, (clip * 32768.0).astype("int16").tobytes())
            log.info("saved %.0f s of %s as %s", len(clip) / SAMPLE_RATE, label, dst)
    for s in transcript.segments:
        s.speaker = names.get(s.speaker, s.speaker)
    saved.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1))
    for path in [workdir / "notes.md", *(Path(x) for x in links if x.endswith(".md"))]:
        if path.exists():
            text = path.read_text()
            for label, name in names.items():
                text = re.sub(rf"\b{re.escape(label)}\b", name, text)
            path.write_text(text)

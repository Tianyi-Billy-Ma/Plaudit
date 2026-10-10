from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str = ""


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None

    def as_text(self) -> str:
        return "\n".join(f"[{_ts(s.start)}] {s.speaker}: {s.text}" for s in self.segments if s.text)

    def to_dict(self) -> dict:
        return asdict(self)


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

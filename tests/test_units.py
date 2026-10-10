from plaudit.diarize import merge_turns, split_turns
from plaudit.models import Segment
from plaudit.outputs import md_to_blocks, safe
from plaudit.sources import parse_details, parse_file_ids, parse_url
from plaudit.summarize import chunk

PLAUD_FILES = """
Files on this page: 2

  ID                        NAME                          DATE              DURATION
  ───────────────────────────────────────────────────────────────────────────────────
  6a1f2c3d4e5f6a7b8c9d0e1f  Lab meeting  with  Bo        2026-10-09 14:00  00:42:10
  0f9e8d7c6b5a4f3e2d1c0b9a  咖啡闲聊                      2026-10-08 09:10  00:05:01

Page 1
"""


def test_parse_plaud_files():
    assert parse_file_ids(PLAUD_FILES) == ["6a1f2c3d4e5f6a7b8c9d0e1f", "0f9e8d7c6b5a4f3e2d1c0b9a"]


def test_parse_plaud_details_and_url():
    d = parse_details("\nFile Details:\n\n  id:           abc\n  name:         Lab meeting\n  start_at:     -\n  created_at:   2026-10-09T14:00:00Z\n")
    assert d == {"id": "abc", "name": "Lab meeting", "created_at": "2026-10-09T14:00:00Z"}
    assert parse_url("\nAudio Download URL:\n\nhttps://x.plaud.ai/a.mp3?sig=1\n\nNote: expires\n") == "https://x.plaud.ai/a.mp3?sig=1"
    assert parse_url("Audio not available for this recording.") is None


def test_merge_and_split_turns():
    turns = [Segment(0, 1, "Speaker 1"), Segment(1.2, 70, "Speaker 1"), Segment(70, 70.1, "Speaker 2")]
    merged = merge_turns(turns)
    assert [(t.start, t.end) for t in merged] == [(0, 70), (70, 70.1)]
    out = split_turns(merged, 30)
    assert [(t.start, t.end) for t in out] == [(0, 30), (30, 60), (60, 70)]


def test_chunk_keeps_lines_whole():
    text = "\n".join(f"line {i}" for i in range(100))
    parts = chunk(text, 100)
    assert "".join(parts) == text + "\n" and all(len(p) <= 100 for p in parts)


def test_md_to_blocks_and_safe():
    types = [b["type"] for b in md_to_blocks("## 摘要\n- a\n1. b\nplain\n### sub")]
    assert types == ["heading_2", "bulleted_list_item", "numbered_list_item", "paragraph", "heading_3"]
    assert safe("a/b:c 会议") == "a_b_c 会议"


def test_stuck_job_gets_stub_page(cfg, clip, tmp_path):
    from datetime import datetime, timedelta

    from plaudit.daemon import Queue
    from plaudit.pipeline import Pipeline
    from plaudit.sources import new_job

    class Boom:
        def diarize(self, wav):
            raise RuntimeError("down")

    class StubSink:
        def __init__(self):
            self.calls = []

        def publish(self, *a, **k):
            return "x"

        def publish_stuck(self, meta, error):
            self.calls.append((meta["title"], error))
            return "notion://stub"

    sink = StubSink()
    q = Queue(cfg)
    job = new_job(q.data / "inbox", clip, "old failure", "manual")
    job["first_failure"] = (datetime.now() - timedelta(hours=25)).isoformat(timespec="seconds")
    q.process(job, Pipeline(cfg, diarizer=Boom(), transcriber=None, summarizer=None, sinks=[sink]))
    assert job["status"] == "stuck" and job["stuck_page"] == "notion://stub"
    assert sink.calls == [("old failure", "RuntimeError: down")]

from pathlib import Path

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

    from conftest import FakeSummarizer, FakeTranscriber

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
    q.process(job, Pipeline(cfg, diarizer=Boom(), transcriber=FakeTranscriber(), summarizer=FakeSummarizer(), sinks=[sink]))
    assert job["status"] == "stuck" and job["stuck_page"] == "notion://stub"
    assert sink.calls == [("old failure", "RuntimeError: down")]


def test_notion_props_follow_the_database_schema(monkeypatch):
    from plaudit import outputs

    schema = {"Name": "title", "Date": "rich_text", "Status": "status", "Page Type": "multi_select", "Source": "select", "PARA": "select"}
    monkeypatch.setattr(outputs, "get_json", lambda url, headers=None: {"properties": {k: {"type": t} for k, t in schema.items()}})
    monkeypatch.setenv("NT", "x")
    out = outputs.NotionOutput({"outputs": {"notion": {"database_id": "db", "token_env": "NT", "status_done": "Done", "status_stuck": "WIP"}}})
    props = out._props({"title": "m", "date": "2026-10-05", "source": "plaud", "source_id": "of_1", "duration_min": 14.1}, "done")
    assert props["Date"] == {"rich_text": [{"text": {"content": "2026-10-05"}}]}
    assert props["Status"] == {"status": {"name": "Done"}}
    assert props["Page Type"] == {"multi_select": [{"name": "Meeting"}, {"name": "Audio"}]}
    assert props["PARA"] == {"select": {"name": "Resource"}}
    assert props["Source"] == {"select": {"name": "Plaudit"}}
    assert "Duration (min)" not in props
    assert out._props({"title": "m", "date": "d", "source": "plaud"}, "stuck")["Status"] == {"status": {"name": "WIP"}}


def test_local_date_converts_plaud_utc_timestamps(monkeypatch):
    import time

    from plaudit.sources import local_date

    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    assert local_date("2026-10-08T20:01:01.781000") == "2026-10-08"  # 16:01 EDT
    assert local_date("2026-10-07T02:32:12") == "2026-10-06"  # 22:32 EDT the evening before
    assert local_date("2026-10-07T02:32:12Z") == "2026-10-06"
    assert local_date("2026-10-05") == "2026-10-05"


def test_folder_pull_yields_each_job_as_its_file_lands(cfg, tmp_path):
    import os

    from plaudit.sources import FolderSource

    inbox = tmp_path / "data" / "inbox"
    inbox.mkdir(parents=True)
    folder = tmp_path / "inbox"
    folder.mkdir()
    for name in ("a.wav", "b.wav"):
        (folder / name).write_bytes(b"x")
        os.utime(folder / name, (0, 0))
    gen = FolderSource(cfg, inbox).pull()
    first = next(gen)  # the first job exists before the second file is touched
    assert first["title"] == "a" and (folder / "b.wav").exists() and not (folder / "a.wav").exists()
    assert next(gen)["title"] == "b"


def test_folder_date_is_local_and_ids_are_unique(cfg, tmp_path, monkeypatch):
    import os
    import time
    from datetime import datetime

    from plaudit.sources import FolderSource

    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    watched = Path(cfg["sources"]["folder"]["path"])
    watched.mkdir(parents=True, exist_ok=True)
    late = datetime(2026, 10, 8, 23, 0).timestamp()  # 23:00 local = 15:00 UTC, still 2026-10-08
    for name in ("Meeting.mp3", "Meeting.wav"):
        (watched / name).write_bytes(b"x")
        os.utime(watched / name, (late, late))
    (tmp_path / "queue-inbox").mkdir()
    jobs = list(FolderSource(cfg, tmp_path / "queue-inbox").pull())
    assert [j["date"] for j in jobs] == ["2026-10-08", "2026-10-08"]
    assert jobs[0]["id"] != jobs[1]["id"]


def test_markdown_output_never_overwrites(cfg, tmp_path):
    from plaudit.models import Transcript
    from plaudit.outputs import MarkdownOutput

    out = MarkdownOutput(cfg)
    t = Transcript([], "zh")
    meta = {"date": "2026-10-11", "title": "Lab", "duration_min": 1, "source": "manual"}
    a = out.publish({**meta, "id": "a"}, "first", t)
    b = out.publish({**meta, "id": "b"}, "second", t)
    assert a != b and "first" in Path(a).read_text() and "second" in Path(b).read_text()


def test_config_allows_notion_only(tmp_path):
    from plaudit import config

    p = tmp_path / "c.yaml"
    p.write_text("outputs:\n  markdown: null\n  notion:\n    database_id: abc\n")
    assert config.load(p)["outputs"]["markdown"] is None

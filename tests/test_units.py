import pytest
from pathlib import Path

from plaudit.diarize import merge_turns, split_turns, stream_chunks
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
    job = new_job(clip, "old failure", "manual")
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


def test_plaud_pull_filters_on_upload_time_and_skips_test_clips(cfg, tmp_path, monkeypatch):
    from datetime import date

    from plaudit import sources
    from plaudit.sources import PlaudSource, seconds

    assert (seconds("2s"), seconds("23m54s"), seconds("1h28m")) == (2, 1434, 5280)
    today = date.today().isoformat()
    files = {"clip00": f"  name: t\n  created_at: {today}T19:23:56\n  start_at: 2026-09-16T18:07:01\n  duration: 2s\n",
             "old000": "  name: o\n  created_at: 2026-06-29T10:00:00\n  start_at: 2026-06-29T09:00:00\n  duration: 1h01m\n",
             "meeting": f"  name: m\n  created_at: {today}T01:00:00\n  start_at: 2026-09-17T00:00:00\n  duration: 23m54s\n"}

    def fake_plaud(self, cmd, *args):
        if cmd == "files":
            return "  ID  NAME\n" + "".join(f"  {k}  x\n" for k in files)
        if cmd == "file":
            return files[args[0]]
        return "https://x.plaud.ai/a.mp3"

    monkeypatch.setattr(PlaudSource, "_plaud", fake_plaud)
    monkeypatch.setattr(sources, "download", lambda url, dest: Path(dest).write_bytes(b"x"))
    (tmp_path / "inbox").mkdir(exist_ok=True)
    src = PlaudSource(cfg, tmp_path / "inbox", tmp_path)
    jobs = list(src.pull())
    assert [j["source_id"] for j in jobs] == ["meeting"]
    assert src.seen == {"clip00", "old000", "meeting"}  # skipped ones are never fetched again


def test_stream_chunks_cover_the_audio_with_overlap():
    # first chunk 10 samples, later ones 12, cursor advances 8, later chunks start 2 samples early
    assert stream_chunks(30, 10, 12, 8, 2) == [(0, 10, False), (6, 18, False), (14, 26, False), (22, 30, True)]
    assert stream_chunks(10, 10, 12, 8, 2) == [(0, 10, True)]  # exactly one chunk: it is the last
    assert stream_chunks(3, 10, 12, 8, 2) == [(0, 3, True)]


def test_failed_job_backs_off_before_retrying(cfg, clip):
    from datetime import datetime, timedelta

    from conftest import FakeSummarizer, FakeTranscriber

    from plaudit.daemon import Queue, due
    from plaudit.pipeline import Pipeline
    from plaudit.sources import new_job

    class Boom:
        def diarize(self, wav):
            raise RuntimeError("down")

    q = Queue(cfg)
    job = new_job(clip, "flaky", "manual")
    pipe = Pipeline(cfg, diarizer=Boom(), transcriber=FakeTranscriber(), summarizer=FakeSummarizer(), sinks=[])
    assert due(job)
    q.process(job, pipe)
    assert job["status"] == "waiting" and not due(job)
    assert timedelta(minutes=9) < datetime.fromisoformat(job["not_before"]) - datetime.now() <= timedelta(minutes=10)
    job["attempts"] = 9
    q.process(job, pipe)
    assert datetime.fromisoformat(job["not_before"]) - datetime.now() <= timedelta(hours=2)
    job["not_before"] = "2000-01-01T00:00:00"
    assert due(job)


def test_long_transcripts_merge_in_rounds_no_call_over_chunk_size(cfg):
    from plaudit.summarize import Summarizer

    cfg["summary"].update(base_url="http://x", chunk_chars=300, language="en")
    s = Summarizer(cfg)
    calls = []

    def fake_chat(system, user):
        calls.append(len(user))
        return "## Summary\n- part %d\n" % len(calls)

    s._chat = fake_chat
    out = s.summarize("".join("line %03d " % i + "x" * 40 + "\n" for i in range(60)))
    assert out.startswith("## Summary") and len(calls) > 2
    assert max(calls) <= 300 + len(s.merge_prompt)
    assert "at most 100 characters" in s.merge_prompt


def test_retry_reuses_saved_transcript(cfg, clip, tmp_path):
    from conftest import FakeSummarizer, FakeTranscriber

    from plaudit.pipeline import Pipeline

    class Boom:
        def diarize(self, wav):
            raise AssertionError("diarizer must not run again")

    workdir = tmp_path / "w"
    workdir.mkdir()
    (workdir / "transcript.json").write_text('{"segments": [{"start": 0, "end": 1, "speaker": "S1", "text": "hi"}], "language": "en"}')
    r = Pipeline(cfg, diarizer=Boom(), transcriber=FakeTranscriber(), summarizer=FakeSummarizer(), sinks=[]).process(clip, {"title": "t", "date": "2026-10-14", "source": "manual"}, workdir)
    assert r["transcript"].as_text() == "[00:00:00] S1: hi"


def test_empty_answer_retries_as_halves(cfg):
    from plaudit.summarize import EmptySummary, Summarizer

    cfg["summary"].update(base_url="http://x", chunk_chars=5000, language="en")
    s = Summarizer(cfg)
    calls = []

    def fake_chat(system, user):
        calls.append(len(user))
        if len(user) > 1500:
            raise EmptySummary("length")
        return "## Summary\n- ok\n"

    s._chat = fake_chat
    text = "".join("line %03d " % i + "x" * 40 + "\n" for i in range(80))  # ~4000 chars, one chunk
    assert s.summarize(text).startswith("## Summary")
    assert calls[0] > 1500 and len(calls) >= 5  # whole, two halves (one more split), merges
    with pytest.raises(EmptySummary):
        s._chat = lambda system, user: (_ for _ in ()).throw(EmptySummary("length"))
        s.summarize(text)

    def merge_fails(system, user):  # halves answer, every merge comes back empty: keep the halves
        if user.startswith(s.merge_prompt) or len(user) > 1500:
            raise EmptySummary("length")
        return "## Summary\n- %d\n" % len(user)

    s._chat = merge_fails
    assert s.summarize(text).count("## Summary") >= 2


def test_summary_answers_are_cached_on_disk(cfg, tmp_path):
    from plaudit.summarize import Summarizer

    cfg["summary"].update(base_url="http://x", chunk_chars=300, language="en")
    s = Summarizer(cfg)
    n = [0]

    def fake(system, user):
        n[0] += 1
        return "## Summary\n- %d\n" % n[0]

    s._chat_uncached = fake
    text = "".join("line %02d " % i + "x" * 40 + "\n" for i in range(20))
    first = s.summarize(text, cache_file=tmp_path / "c.json")
    calls = n[0]
    assert calls > 1 and (tmp_path / "c.json").exists()
    s2 = Summarizer(cfg)
    s2._chat_uncached = lambda system, user: (_ for _ in ()).throw(AssertionError("must hit cache"))
    assert s2.summarize(text, cache_file=tmp_path / "c.json") == first


def test_cut_off_reply_counts_as_empty_answer(cfg, monkeypatch):
    import http.client

    from plaudit import summarize
    from plaudit.summarize import EmptySummary, Summarizer

    cfg["summary"].update(base_url="http://x", language="en")
    monkeypatch.setattr(summarize, "post_json", lambda *a, **k: (_ for _ in ()).throw(http.client.IncompleteRead(b"")))
    with pytest.raises(EmptySummary):
        Summarizer(cfg)._chat_uncached("s", "u")


def test_reasoning_effort_is_sent_only_when_configured(cfg, monkeypatch):
    from plaudit import summarize
    from plaudit.summarize import Summarizer

    sent = []
    monkeypatch.setattr(summarize, "post_json", lambda url, payload, *a, **k: sent.append(payload) or {"choices": [{"message": {"content": "ok"}}]})
    cfg["summary"].update(base_url="http://x")
    Summarizer(cfg)._chat_uncached("s", "u")
    cfg["summary"]["reasoning_effort"] = "none"
    Summarizer(cfg)._chat_uncached("s", "u")
    assert "reasoning_effort" not in sent[0] and sent[1]["reasoning_effort"] == "none"


def test_dotenv_next_to_config_fills_missing_env_vars(tmp_path, monkeypatch):
    from plaudit import config

    monkeypatch.delenv("PLAUDIT_T_A", raising=False)
    monkeypatch.setenv("PLAUDIT_T_B", "from-env")
    (tmp_path / "plaudit.yaml").write_text("retention_days: 3\n")
    (tmp_path / ".env").write_text("# secrets\nexport PLAUDIT_T_A='a b'\nPLAUDIT_T_B=\"from-file\"\nbroken line\n")
    assert config.load(tmp_path / "plaudit.yaml")["retention_days"] == 3
    assert config.secret("PLAUDIT_T_A") == "a b"
    assert config.secret("PLAUDIT_T_B") == "from-env"  # the environment wins over the file


def test_process_command_output_joins_the_queue_as_done(cfg, clip, tmp_path, monkeypatch):
    from conftest import FakeDiarizer, FakeTranscriber

    from plaudit import cli, pipeline
    from plaudit.daemon import Queue

    monkeypatch.setattr(pipeline, "diarize", type("D", (), {"build": staticmethod(lambda cfg: FakeDiarizer())}))
    monkeypatch.setattr(pipeline, "transcribe", type("T", (), {"build": staticmethod(lambda cfg: FakeTranscriber())}))
    assert cli.main(["-c", str(tmp_path / "plaudit.yaml"), "process", str(clip), "--title", "Manual"]) == 0
    jobs = Queue(cfg).jobs()
    assert len(jobs) == 1 and jobs[0]["status"] == "done" and jobs[0]["title"] == "Manual"
    assert (Queue(cfg).done_dir / jobs[0]["id"] / clip.name).exists()  # the original audio is kept for the backup


def test_webhook_output_formats_and_gets_earlier_links(cfg, clip, tmp_path, monkeypatch):
    from conftest import FakeDiarizer, FakeSummarizer, FakeTranscriber

    from plaudit import outputs
    from plaudit.pipeline import Pipeline

    sent = []
    monkeypatch.setattr(outputs, "post_json", lambda url, payload, *a, **k: sent.append((url, payload)) or {})
    cfg["outputs"]["webhook"].update(url="https://hooks.example/x", format="slack")
    pipe = Pipeline(cfg, diarizer=FakeDiarizer(), transcriber=FakeTranscriber(), summarizer=FakeSummarizer(),
                    sinks=[outputs.MarkdownOutput(cfg), outputs.WebhookOutput(cfg)])
    r = pipe.process(clip, {"title": "Standup", "date": "2026-10-20", "source": "manual", "id": "j1"}, tmp_path / "w")
    url, payload = sent[0]
    assert url == "https://hooks.example/x" and payload["text"].startswith("*Standup (2026-10-20, 0.1 min)\n")
    assert r["links"][0] in payload["text"] and "## 摘要" in payload["text"]  # the Markdown path, then the minutes

    cfg["outputs"]["webhook"]["format"] = "generic"
    outputs.WebhookOutput(cfg).publish_stuck({"title": "t", "date": "d"}, "boom")
    assert sent[1][1] == {"event": "meeting.stuck", "meta": {"title": "t", "date": "d"}, "error": "boom"}

    monkeypatch.setenv("PLAUDIT_WEBHOOK_URL", "https://hooks.example/env")
    cfg["outputs"]["webhook"]["url"] = None
    assert [type(o).__name__ for o in outputs.build(cfg)] == ["MarkdownOutput", "WebhookOutput"]
    assert outputs.build(cfg)[1].url == "https://hooks.example/env"


def test_email_output(cfg, monkeypatch):
    from plaudit import outputs
    from plaudit.models import Transcript

    class FakeSMTP:
        calls = []

        def __init__(self, host, port, timeout=0):
            self.calls.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def starttls(self):
            self.calls.append(("starttls",))

        def login(self, user, password):
            self.calls.append(("login", user, password))

        def send_message(self, msg):
            self.calls.append(("send", msg))

    monkeypatch.setattr(outputs.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("PLAUDIT_SMTP_USER", "me@example.com")
    monkeypatch.setenv("PLAUDIT_SMTP_PASSWORD", "pw")
    cfg["outputs"]["email"].update(to="you@example.com", smtp_host="smtp.example.com")
    out = outputs.build(cfg)[-1]
    assert isinstance(out, outputs.EmailOutput)
    t = Transcript([Segment(0, 1, "A", "hi")])
    assert out.publish({"title": "Standup", "date": "2026-10-20", "source": "manual", "duration_min": 1}, "## Summary\n- x", t,
                       links=["https://notion.so/p"]) == "mailto:you@example.com"
    kinds = [c[0] for c in FakeSMTP.calls]
    assert kinds == ["connect", "starttls", "login", "send"] and FakeSMTP.calls[0][1:] == ("smtp.example.com", 587)
    msg = FakeSMTP.calls[-1][1]
    assert msg["Subject"] == "[Plaudit] Standup (2026-10-20)" and msg["From"] == "me@example.com" and msg["To"] == "you@example.com"
    body, attachment = msg.iter_parts()
    assert "https://notion.so/p" in body.get_content() and "- x" in body.get_content()
    assert attachment.get_filename() == "notes.md" and "[00:00:00] A: hi" in attachment.get_content()


def test_http_transcriber_sends_iso_language_code(cfg, monkeypatch):
    from plaudit import transcribe

    cfg["transcription"].update(backend="openai", base_url="http://x/v1")
    cfg["language"] = "Chinese"
    seen = []
    monkeypatch.setattr(transcribe, "post_multipart", lambda url, fields, *a, **k: seen.append(fields) or {"text": " 你好 "})
    monkeypatch.setattr(transcribe, "read_pcm", lambda wav: b"\0" * 32000)
    turns = transcribe.build(cfg).transcribe(Path("x.wav"), [Segment(0, 1, "Speaker 1")])
    assert seen[0]["language"] == "zh" and turns[0].text == "你好"
    cfg["language"] = None
    transcribe.build(cfg).transcribe(Path("x.wav"), [Segment(0, 1, "Speaker 1")])
    assert "language" not in seen[1]

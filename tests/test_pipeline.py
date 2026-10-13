import json
import shutil
from pathlib import Path

from conftest import FakeDiarizer, FakeSummarizer, FakeTranscriber

import pytest

from plaudit.daemon import Queue, mem_available_gb
from plaudit.pipeline import Pipeline


def make(cfg):
    return Pipeline(cfg, diarizer=FakeDiarizer(), transcriber=FakeTranscriber(), summarizer=FakeSummarizer())


def test_process_writes_markdown(cfg, clip, tmp_path):
    meta = {"title": "Weekly sync", "date": "2026-10-10", "source": "manual"}
    result = make(cfg).process(clip, meta, tmp_path / "work")
    assert result["meta"]["duration_min"] == 0.1
    note = Path(result["links"][0]).read_text()
    assert "# Weekly sync" in note and "## 摘要" in note and "[00:00:02] Speaker 2: turn 1" in note
    saved = json.loads((tmp_path / "work" / "transcript.json").read_text())
    assert [s["speaker"] for s in saved["segments"]] == ["Speaker 1", "Speaker 2"]
    assert not (tmp_path / "work" / "audio16k.wav").exists()


def test_queue_end_to_end_via_folder(cfg, clip, tmp_path):
    inbox = Path(cfg["sources"]["folder"]["path"])
    inbox.mkdir()
    shutil.copy(clip, inbox / "standup.m4a.wav")
    import os
    os.utime(inbox / "standup.m4a.wav", (0, 0))  # old enough to count as fully written
    q = Queue(cfg)
    q.run_once(make(cfg))
    jobs = q.jobs()
    assert len(jobs) == 1 and jobs[0]["status"] == "done", jobs
    assert (q.done_dir / jobs[0]["id"] / "notes.md").exists()
    assert not list(inbox.iterdir())


def test_failed_job_stays_waiting(cfg, clip, tmp_path):
    q = Queue(cfg)
    from plaudit.sources import new_job
    job = new_job(q.data / "inbox", clip, "bad", "manual")

    class Boom:
        def diarize(self, wav):
            raise RuntimeError("gpu on fire")

    q.process(job, Pipeline(cfg, diarizer=Boom(), transcriber=FakeTranscriber(), summarizer=FakeSummarizer()))
    assert job["status"] == "waiting" and job["attempts"] == 1 and "gpu on fire" in job["last_error"]


def test_job_left_running_by_a_killed_run_counts_as_an_attempt(cfg, tmp_path, monkeypatch):
    q = Queue(cfg)
    (q.data / "inbox").mkdir()
    Path(cfg["sources"]["folder"]["path"]).mkdir(exist_ok=True)
    q.save({"id": "k", "title": "k", "status": "running", "attempts": 2})
    monkeypatch.setattr(Queue, "process", lambda self, job, pipeline: pytest.fail("must not run a stuck job"))
    q.run_once()
    job = q.jobs()[0]
    assert job["status"] == "stuck" and job["attempts"] == 3 and "died" in job["last_error"]


def test_mem_available():
    assert round(mem_available_gb("MemTotal: 1 kB\nMemAvailable:  8388608 kB\n"), 1) == 8.0


def test_retire_backs_up_then_deletes_after_retention(cfg, tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    from plaudit import daemon

    cfg["backup"]["remote"] = "remote:Meetings"
    q = Queue(cfg)
    old = (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")
    for jid, finished in (("a-old", old), ("b-new", datetime.now().isoformat(timespec="seconds"))):
        (q.done_dir / jid).mkdir()
        q.save({"id": jid, "title": jid, "date": "2026-10-01", "status": "done", "finished_at": finished})
    calls = []
    monkeypatch.setattr(daemon, "backup_dir", lambda local, remote, date: calls.append(local.name) or f"{remote}/2026/10/{local.name}")
    q.retire()
    by_id = {j["id"]: j for j in q.jobs()}
    assert calls == ["a-old", "b-new"]  # a done job without a backup gets one
    assert by_id["a-old"]["status"] == "retired" and not (q.done_dir / "a-old").exists()
    assert by_id["b-new"]["status"] == "done" and (q.done_dir / "b-new").exists()

    def boom(local, remote, date):
        raise RuntimeError("drive down")

    (q.done_dir / "c").mkdir()
    q.save({"id": "c", "title": "c", "date": "2026-10-01", "status": "done", "finished_at": old})
    monkeypatch.setattr(daemon, "backup_dir", boom)
    q.retire()
    c = {j["id"]: j for j in q.jobs()}["c"]
    assert c["status"] == "done" and "backup" not in c and (q.done_dir / "c").exists()  # never delete without a backup


def test_corrupt_job_file_is_skipped_and_save_is_atomic(cfg, tmp_path):
    q = Queue(cfg)
    (q.jobs_dir / "bad.json").write_text('{"id": "trunc')
    q.save({"id": "good", "status": "waiting"})
    assert [j["id"] for j in q.jobs()] == ["good"]
    assert not list(q.jobs_dir.glob("*.tmp"))

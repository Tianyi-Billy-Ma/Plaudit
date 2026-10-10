import json
import shutil
from pathlib import Path

from conftest import FakeDiarizer, FakeSummarizer, FakeTranscriber

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


def test_mem_available():
    assert round(mem_available_gb("MemTotal: 1 kB\nMemAvailable:  8388608 kB\n"), 1) == 8.0

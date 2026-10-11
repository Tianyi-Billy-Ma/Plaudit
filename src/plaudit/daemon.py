"""The queue: pull sources, process waiting jobs one at a time behind a lock and a memory guard, back up, retire."""
from __future__ import annotations

import fcntl
import json
import logging
import shutil
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from . import sources
from .backup import backup_dir
from .pipeline import Pipeline

log = logging.getLogger(__name__)
STUCK_AFTER = timedelta(hours=24)


def mem_available_gb(meminfo: str | None = None) -> float:
    text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return float("inf")


class Queue:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data = Path(cfg["data_dir"])
        self.jobs_dir = self.data / "jobs"
        self.done_dir = self.data / "done"
        for d in (self.jobs_dir, self.done_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, job: dict) -> None:
        tmp = self.jobs_dir / f"{job['id']}.json.tmp"
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1))
        tmp.replace(self.jobs_dir / f"{job['id']}.json")  # atomic: a kill mid-write never leaves a half file

    def jobs(self) -> list[dict]:
        out = []
        for p in self.jobs_dir.glob("*.json"):
            try:
                out.append(json.loads(p.read_text()))
            except ValueError:
                log.error("skipping unreadable job file %s", p)
        return sorted(out, key=lambda j: j["id"])

    def pull(self) -> int:
        n = 0
        for src in sources.build(self.cfg, self.data):
            try:
                for job in src.pull():
                    self.save(job)
                    n += 1
            except Exception:
                log.exception("source %s failed", type(src).__name__)
        return n

    @contextmanager
    def lock(self):
        """One pipeline at a time per data dir; yields False when another run holds the lock."""
        with open(self.data / "plaudit.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log.info("another plaudit run holds the lock; skipping")
                yield False
                return
            yield True

    def run_once(self, pipeline: Pipeline | None = None) -> None:
        with self.lock() as got:
            if not got:
                return
            log.info("pulled %d new recording(s)", self.pull())
            for job in self.jobs():
                if job["status"] not in ("waiting",):
                    continue
                free = mem_available_gb()
                if pipeline is None and free < self.cfg["min_free_mem_gb"]:  # once loaded, the models are the usage
                    log.warning("only %.1f GB free, below min_free_mem_gb; jobs stay queued", free)
                    break
                pipeline = pipeline or Pipeline(self.cfg)
                self.process(job, pipeline)
            self.retire()

    def process(self, job: dict, pipeline: Pipeline) -> None:
        workdir = self.done_dir / job["id"]
        try:
            result = pipeline.process(Path(job["audio"]), job, workdir)
            shutil.move(job["audio"], workdir / Path(job["audio"]).name)
            job.update(status="done", links=result["links"], finished_at=datetime.now().isoformat(timespec="seconds"),
                       duration_min=result["meta"]["duration_min"])
            log.info("%s done: %s", job["title"], result["links"])  # backup happens in retire(), same run
        except Exception as e:
            job["attempts"] += 1
            job["last_error"] = f"{type(e).__name__}: {e}"[:500]
            job.setdefault("first_failure", datetime.now().isoformat(timespec="seconds"))
            if datetime.now() - datetime.fromisoformat(job["first_failure"]) > STUCK_AFTER and job["status"] != "stuck":
                job["status"] = "stuck"
                for sink in pipeline.sinks:
                    if hasattr(sink, "publish_stuck"):
                        try:
                            job["stuck_page"] = sink.publish_stuck(job, job["last_error"])
                        except Exception:
                            log.exception("could not publish stuck page for %s", job["title"])
            log.exception("%s failed (attempt %d, status %s)", job["title"], job["attempts"], job["status"])
        self.save(job)

    def retire(self) -> None:
        """Back up done jobs that still lack a backup, then delete local copies only after a verified backup
        and the retention period."""
        cutoff = datetime.now() - timedelta(days=self.cfg["retention_days"])
        remote = self.cfg["backup"]["remote"]
        for job in self.jobs():
            if job["status"] != "done":
                continue
            if remote and not job.get("backup"):
                try:
                    job["backup"] = backup_dir(self.done_dir / job["id"], remote, job["date"])
                    self.save(job)
                except Exception:
                    log.exception("backup of %s failed; will retry next run", job["title"])
                    continue
            if job.get("backup") and datetime.fromisoformat(job["finished_at"]) < cutoff:
                shutil.rmtree(self.done_dir / job["id"], ignore_errors=True)
                job["status"] = "retired"
                self.save(job)

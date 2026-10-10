"""Where recordings come from. Each source's pull() drops new audio into data/inbox and returns job dicts."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".webm", ".wma"}


def new_job(inbox: Path, audio: Path, title: str, source: str, source_id: str = "", when: str | None = None) -> dict:
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"\W+", "", source_id or title)[:24]
    return {"id": jid, "title": title, "date": (when or date.today().isoformat())[:10], "source": source,
            "source_id": source_id, "audio": str(audio), "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "waiting", "attempts": 0}


class FolderSource:
    """A watched folder: any audio file dropped in is moved to the inbox and queued."""

    def __init__(self, cfg: dict, inbox: Path):
        self.dir = Path(cfg["sources"]["folder"]["path"])
        self.inbox = inbox

    def pull(self) -> list[dict]:
        jobs = []
        if not self.dir.exists():
            return jobs
        for f in sorted(self.dir.iterdir()):
            if f.suffix.lower() not in AUDIO_EXT or f.name.startswith("."):
                continue
            if time.time() - f.stat().st_mtime < 10:  # still being written
                continue
            when = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            dest = self.inbox / f"{int(time.time())}-{f.name}"
            shutil.move(str(f), dest)
            jobs.append(new_job(self.inbox, dest, f.stem, "folder", when=when))
        return jobs


class PlaudSource:
    """Official @plaud-ai/cli. Plaud has no consumer webhook, so this runs on a timer and keeps seen.json."""

    def __init__(self, cfg: dict, inbox: Path, state_dir: Path):
        self.lookback = int(cfg["sources"]["plaud"]["lookback_days"])
        self.inbox = inbox
        self.seen_path = state_dir / "seen.json"
        self.seen: set[str] = set(json.loads(self.seen_path.read_text())) if self.seen_path.exists() else set()

    def _plaud(self, *args: str) -> str:
        env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
        return subprocess.run(["plaud", *args], check=True, capture_output=True, text=True, env=env, timeout=120).stdout

    def pull(self) -> list[dict]:
        since = (date.today() - timedelta(days=self.lookback)).isoformat()
        ids = parse_file_ids(self._plaud("files", "-s", "100", "--date-from", since))
        jobs = []
        for fid in ids:
            if fid in self.seen:
                continue
            details = parse_details(self._plaud("file", fid))
            url = parse_url(self._plaud("audio", fid))
            if not url:
                log.info("plaud %s: audio not available yet", fid)
                continue
            ext = Path(url.split("?")[0]).suffix or ".mp3"
            dest = self.inbox / f"plaud-{fid}{ext}"
            urllib.request.urlretrieve(url, dest)
            self.seen.add(fid)
            self.seen_path.write_text(json.dumps(sorted(self.seen)))
            jobs.append(new_job(self.inbox, dest, details.get("name") or f"Plaud {fid}", "plaud", fid,
                                details.get("start_at") or details.get("created_at")))
        return jobs


# shortcut: `plaud files` has no --json flag, so we read its table; upgrade when the CLI grows one
def parse_file_ids(table: str) -> list[str]:
    ids = []
    for line in table.splitlines():
        m = re.match(r"^\s{2}([A-Za-z0-9_-]{6,})\s{2,}", line)
        if m and m.group(1) != "ID":
            ids.append(m.group(1))
    return ids


def parse_details(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s+(\w+):\s+(.*\S)\s*$", line)
        if m and m.group(2) != "-":
            out[m.group(1)] = m.group(2)
    return out


def parse_url(text: str) -> str | None:
    m = re.search(r"https?://\S+", text)
    return m.group(0) if m else None


def build(cfg: dict, data_dir: Path) -> list:
    inbox = data_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    srcs = [FolderSource(cfg, inbox)]
    if cfg["sources"]["plaud"].get("enabled"):
        srcs.append(PlaudSource(cfg, inbox, data_dir))
    return srcs

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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".webm", ".wma"}


def new_job(audio: Path, title: str, source: str, source_id: str = "", when: str | None = None) -> dict:
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"\W+", "", source_id or title)[:24]
    return {"id": jid, "title": title, "date": local_date(when), "source": source,
            "source_id": source_id, "audio": str(audio), "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "waiting", "attempts": 0}


def local_date(when: str | None) -> str:
    """YYYY-MM-DD in local time. A bare date stays as is; a timestamp without zone is UTC (what Plaud returns)."""
    if not when:
        return date.today().isoformat()
    if len(when) <= 10:
        return when[:10]
    dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone().date().isoformat()


class FolderSource:
    """A watched folder: any audio file dropped in is moved to the inbox and queued."""

    def __init__(self, cfg: dict, inbox: Path):
        self.dir = Path(cfg["sources"]["folder"]["path"])
        self.inbox = inbox

    def pull(self) -> Iterator[dict]:
        """A generator, so the queue saves each job as soon as its file is in place."""
        if not self.dir.exists():
            return
        for f in sorted(self.dir.iterdir()):
            if f.suffix.lower() not in AUDIO_EXT or f.name.startswith("."):
                continue
            if time.time() - f.stat().st_mtime < 10:  # still being written
                continue
            when = datetime.fromtimestamp(f.stat().st_mtime).astimezone().isoformat()  # local, not UTC like Plaud
            dest = self.inbox / f"{int(time.time())}-{f.name}"
            shutil.move(str(f), dest)
            yield new_job(dest, f.stem, "folder", source_id=dest.name, when=when)


class RcloneSource:
    """Any rclone remote (Dropbox, Google Drive, OneDrive, S3, Nextcloud...) used as a drop folder: new audio files
    are moved into the watched folder, which queues them. Moving keeps the remote as the inbox it is; the finished
    recording is backed up by `backup.remote`."""

    def __init__(self, cfg: dict):
        self.remote = cfg["sources"]["rclone"]["remote"]
        self.folder = cfg["sources"]["folder"]["path"]

    def pull(self) -> Iterator[dict]:
        include = "*.{" + ",".join(e[1:] for e in sorted(AUDIO_EXT)) + "}"
        subprocess.run(["rclone", "move", "--ignore-case", "--include", include, "--max-depth", "1", self.remote, self.folder],
                       check=True, timeout=1800)
        return iter(())  # FolderSource runs next and queues what arrived


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

    def pull(self) -> Iterator[dict]:
        since = (date.today() - timedelta(days=self.lookback)).isoformat()
        # Filter on upload time, not --date-from: that uses the recorder's clock, which can be weeks off.
        for fid in parse_file_ids(self._plaud("files", "-s", "100")):
            if fid in self.seen:
                continue
            details = parse_details(self._plaud("file", fid))
            if details.get("created_at", "")[:10] < since or seconds(details.get("duration", "")) < 60:
                self._mark_seen(fid)  # too old, or a test clip; never look at it again
                continue
            url = parse_url(self._plaud("audio", fid))
            if not url:
                log.info("plaud %s: audio not available yet", fid)
                continue
            ext = Path(url.split("?")[0]).suffix or ".mp3"
            dest = self.inbox / f"plaud-{fid}{ext}"
            download(url, dest)
            self._mark_seen(fid)
            yield new_job(dest, details.get("name") or f"Plaud {fid}", "plaud", fid,
                          details.get("start_at") or details.get("created_at"))

    def _mark_seen(self, fid: str) -> None:
        self.seen.add(fid)
        self.seen_path.write_text(json.dumps(sorted(self.seen)))


def seconds(duration: str) -> int:
    """Plaud prints durations like 2s, 23m54s, 1h28m."""
    return sum(int(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in re.findall(r"(\d+)([hms])", duration))


def download(url: str, dest: Path, timeout: int = 300) -> None:
    """urlretrieve has no timeout; a stalled download would hold the queue lock forever."""
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


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
    srcs = []
    if cfg["sources"].get("rclone", {}).get("remote"):
        srcs.append(RcloneSource(cfg))  # before the folder source, so its files are queued in the same run
    srcs.append(FolderSource(cfg, inbox))
    if cfg["sources"]["plaud"].get("enabled"):
        srcs.append(PlaudSource(cfg, inbox, data_dir))
    return srcs

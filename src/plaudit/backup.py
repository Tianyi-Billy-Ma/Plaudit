"""rclone backup of a finished job folder. Verified with `rclone check` before anyone deletes anything."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def backup_dir(local: Path, remote: str, date: str) -> str:
    """Copy `local` to <remote>/YYYY/MM/<local.name>/ and verify. Returns the remote path."""
    dest = f"{remote.rstrip('/')}/{date[:4]}/{date[5:7]}/{local.name}"
    subprocess.run(["rclone", "copy", "--checksum", str(local), dest], check=True)
    subprocess.run(["rclone", "check", "--one-way", str(local), dest], check=True)
    return dest

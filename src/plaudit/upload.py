"""`plaudit serve`: a tiny HTTP endpoint that drops an uploaded recording into the watched folder, for phones and
scripts: `curl -T meeting.m4a -H "Authorization: Bearer $PLAUDIT_UPLOAD_TOKEN" http://host:8787/meeting.m4a`."""
from __future__ import annotations

import logging
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .outputs import safe
from .sources import AUDIO_EXT

log = logging.getLogger(__name__)


def make_handler(folder: Path, token: str):
    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return self.send_error(401)
            name = Path(unquote(urlsplit(self.path).path)).name  # the URL path is the file name
            if Path(name).suffix.lower() not in AUDIO_EXT:
                return self.send_error(415, f"file name must end in one of {sorted(AUDIO_EXT)}")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self.send_error(411)
            dest = folder / f"{safe(Path(name).stem)}-{int(time.time())}{Path(name).suffix.lower()}"
            with open(dest, "wb") as f:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            self.send_response(201)
            self.end_headers()
            self.wfile.write(f"queued {dest.name}\n".encode())

        do_POST = do_PUT

        def log_message(self, fmt, *args):
            log.info("%s " + fmt, self.client_address[0], *args)

    return Handler


def serve(cfg: dict, host: str = "0.0.0.0", port: int | None = None) -> None:
    from .config import secret

    uc = cfg["sources"]["upload"]
    token = secret(uc["token_env"])
    if not token:
        raise SystemExit(f"plaudit serve: set ${uc['token_env']} first; the endpoint refuses to run without a token")
    folder = Path(cfg["sources"]["folder"]["path"])
    folder.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((host, port or int(uc["port"])), make_handler(folder, token))
    log.info("upload endpoint on %s:%d, files go to %s", *srv.server_address[:2], folder)
    srv.serve_forever()

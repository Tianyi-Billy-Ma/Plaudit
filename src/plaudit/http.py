"""Tiny JSON HTTP client (stdlib only) for the OpenAI-compatible and Notion APIs."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import __version__

# Cloudflare in front of some endpoints rejects the default urllib agent (error 1010)
UA = f"plaudit/{__version__}"


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 600, retries: int = 3,
              method: str = "POST") -> dict:
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", "User-Agent": UA, **(headers or {})}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise HttpError(e.code, body) from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def get_json(url: str, headers: dict | None = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode(errors="replace")) from None


def post_multipart(url: str, fields: dict[str, str], file_field: str, filename: str, content: bytes,
                   headers: dict | None = None, timeout: int = 600) -> dict:
    boundary = "----plaudit" + str(int(time.time() * 1000))
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
             f"Content-Type: audio/wav\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    hdrs = {"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": UA, **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode(errors="replace")) from None

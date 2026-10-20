"""plaudit.yaml loading. Secrets never live here: they come from environment variables, or from a `.env` file
next to plaudit.yaml (KEY=value lines) for values the environment does not already set."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "data_dir": "~/plaudit/data",
    "glossary": "~/plaudit/glossary.txt",
    "language": None,  # None = auto-detect; or "Chinese" / "English"
    "sources": {
        "folder": {"path": "~/plaudit/inbox"},
        "plaud": {"enabled": False, "lookback_days": 7},
    },
    "diarization": {"backend": "none", "model": "nvidia/Nemotron-3-Diarization", "max_turn_seconds": 30},
    "transcription": {"backend": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B-hf", "base_url": None},
    "summary": {
        "base_url": None,
        "model": "glm-5.3-flash",
        "api_key_env": "OPENAI_API_KEY",
        "language": "zh",
        "chunk_chars": 12000,
        "reasoning_effort": None,  # "none" switches a reasoning model's thinking off (GLM, Qwen3, DeepSeek)
    },
    "outputs": {
        "markdown": {"dir": "~/plaudit/data/notes"},
        "notion": {"database_id": None, "token_env": "NOTION_TOKEN"},
        "webhook": {"url": None, "url_env": "PLAUDIT_WEBHOOK_URL", "format": "generic"},  # generic | slack | discord
        "email": {"to": [], "smtp_host": None, "smtp_port": 587, "tls": "starttls",  # starttls | ssl | none
                  "from": None, "user_env": "PLAUDIT_SMTP_USER", "password_env": "PLAUDIT_SMTP_PASSWORD"},
    },
    "backup": {"remote": None},  # e.g. "gdrive:Meetings"; None = off
    "retention_days": 7,
    "min_free_mem_gb": 8,
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def config_path(arg: str | os.PathLike | None = None) -> Path:
    return Path(arg or os.environ.get("PLAUDIT_CONFIG", "~/plaudit/plaudit.yaml")).expanduser()


def load(path: str | os.PathLike | None = None) -> dict:
    path = config_path(path)
    user = yaml.safe_load(path.read_text()) if path.exists() else {}
    load_env(path.with_name(".env"))
    cfg = _merge(copy.deepcopy(DEFAULTS), user or {})  # nested dicts are shared otherwise and edits leak between loads
    for key in ("data_dir", "glossary"):
        cfg[key] = str(Path(cfg[key]).expanduser())
    cfg["sources"]["folder"]["path"] = str(Path(cfg["sources"]["folder"]["path"]).expanduser())
    if cfg["outputs"].get("markdown"):
        cfg["outputs"]["markdown"]["dir"] = str(Path(cfg["outputs"]["markdown"]["dir"]).expanduser())
    return cfg


def load_env(path: Path) -> None:
    """Put KEY=value lines of `path` into the environment unless already set there. Blank lines, `#` comments,
    an `export ` prefix and single or double quotes around the value are accepted."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def secret(env_name: str | None) -> str | None:
    return os.environ.get(env_name) if env_name else None


def glossary(cfg: dict) -> list[str]:
    p = Path(cfg["glossary"])
    if not p.exists():
        return []
    return [t.strip() for t in p.read_text().splitlines() if t.strip() and not t.startswith("#")]

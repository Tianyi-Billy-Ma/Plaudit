"""plaudit.yaml loading. Secrets never live here: they come from environment variables."""
from __future__ import annotations

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
        "api_key_env": "PLAUDIT_SUMMARY_API_KEY",
        "language": "zh",
        "chunk_chars": 12000,
    },
    "outputs": {
        "markdown": {"dir": "~/plaudit/data/notes"},
        "notion": {"database_id": None, "token_env": "NOTION_TOKEN"},
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


def load(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("PLAUDIT_CONFIG", "~/plaudit/plaudit.yaml")).expanduser()
    user = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = _merge(DEFAULTS, user or {})
    for key in ("data_dir", "glossary"):
        cfg[key] = str(Path(cfg[key]).expanduser())
    cfg["sources"]["folder"]["path"] = str(Path(cfg["sources"]["folder"]["path"]).expanduser())
    if cfg["outputs"].get("markdown"):
        cfg["outputs"]["markdown"]["dir"] = str(Path(cfg["outputs"]["markdown"]["dir"]).expanduser())
    return cfg


def secret(env_name: str | None) -> str | None:
    return os.environ.get(env_name) if env_name else None


def glossary(cfg: dict) -> list[str]:
    p = Path(cfg["glossary"])
    if not p.exists():
        return []
    return [t.strip() for t in p.read_text().splitlines() if t.strip() and not t.startswith("#")]

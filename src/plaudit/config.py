"""plaudit.yaml loading. The config folder is ~/.config/plaudit (plaudit.yaml, .env, glossary.txt, voices/); the
things a user looks at live in visible folders (~/plaudit/inbox, ~/plaudit/notes) and queue state in ~/.local/share.
Secrets never live in plaudit.yaml: they come from environment variables, or from the `.env` file next to it."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "data_dir": None,  # queue state (jobs, finished job folders): $XDG_DATA_HOME/plaudit, i.e. ~/.local/share/plaudit
    "glossary": None,  # glossary.txt next to plaudit.yaml
    "language": None,  # None = auto-detect; or "Chinese" / "English"
    "sources": {
        "folder": {"path": "~/plaudit/inbox"},
        "plaud": {"enabled": False, "lookback_days": 7},
        "rclone": {"remote": None},  # e.g. "dropbox:Recordings": audio there is moved into the watched folder
        "upload": {"port": 8787, "token_env": "PLAUDIT_UPLOAD_TOKEN"},  # `plaudit serve`
    },
    "diarization": {"backend": "none", "model": "nvidia/Nemotron-3-Diarization", "max_turn_seconds": 30},
    # who is Speaker 2: voice samples in `voices` (one audio file per person) are matched against each diarized speaker
    "speakers": {"backend": "redimnet2", "model": "b6", "voices": None, "threshold": None},  # voices/ next to plaudit.yaml
    "transcription": {"backend": "qwen_asr", "model": "Qwen/Qwen3-ASR-1.7B-hf", "provider": None, "base_url": None,
                      "api_key_env": None},
    "summary": {
        "provider": None,  # a PROVIDERS name: fills base_url / api_key_env (/ temperature) unless set explicitly
        "base_url": None,
        "model": "glm-5.3-flash",
        "api_key_env": "OPENAI_API_KEY",
        "language": "zh",
        "chunk_chars": 12000,
        "temperature": 0.2,  # null sends none (Claude 4.7+ rejects values below 1)
        "reasoning_effort": None,  # "none" switches a reasoning model's thinking off (GLM, Qwen3, DeepSeek)
    },
    "outputs": {
        "markdown": {"dir": "~/plaudit/notes"},
        "notion": {"database_id": None, "token_env": "NOTION_TOKEN"},
        "webhook": {"url": None, "url_env": "PLAUDIT_WEBHOOK_URL"},  # JSON POST for n8n, Zapier, Home Assistant
        "notify": {"urls": [], "urls_env": "PLAUDIT_NOTIFY_URLS"},  # Apprise URLs: Slack, Discord, Teams, e-mail...
    },
    "backup": {"remote": None},  # e.g. "gdrive:Meetings"; None = off
    "retention_days": 7,
    "min_free_mem_gb": 8,
}

# OpenAI-compatible endpoints. The hosted ones also need the named key in the environment; the local servers none.
PROVIDERS: dict[str, dict] = {
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "api_key_env": "ANTHROPIC_API_KEY", "temperature": None},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key_env": "GEMINI_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
    "vllm": {"base_url": "http://localhost:8000/v1", "api_key_env": None},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "api_key_env": None},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def xdg(var: str, fallback: str) -> Path:
    return Path(os.environ.get(var) or fallback).expanduser() / "plaudit"


def config_path(arg: str | os.PathLike | None = None) -> Path:
    """--config, else $PLAUDIT_CONFIG, else ~/.config/plaudit/plaudit.yaml; the pre-0.2 ~/plaudit/plaudit.yaml is
    still read when it exists and the new one does not."""
    if arg or os.environ.get("PLAUDIT_CONFIG"):
        return Path(arg or os.environ["PLAUDIT_CONFIG"]).expanduser()
    path, legacy = xdg("XDG_CONFIG_HOME", "~/.config") / "plaudit.yaml", Path("~/plaudit/plaudit.yaml").expanduser()
    return legacy if not path.exists() and legacy.exists() else path


def load(path: str | os.PathLike | None = None) -> dict:
    path = config_path(path)
    user = yaml.safe_load(path.read_text()) if path.exists() else {}
    _refuse_secrets(user or {})
    load_env(path.with_name(".env"))
    cfg = _merge(copy.deepcopy(DEFAULTS), user or {})  # nested dicts are shared otherwise and edits leak between loads
    for section in ("summary", "transcription"):
        if name := cfg[section]["provider"]:
            if name not in PROVIDERS:
                raise ValueError(f"{section}.provider {name!r} unknown; one of: {', '.join(PROVIDERS)}")
            given = (user or {}).get(section) or {}
            cfg[section].update({k: v for k, v in PROVIDERS[name].items() if k in cfg[section] and k not in given})
    cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser() if cfg["data_dir"] else xdg("XDG_DATA_HOME", "~/.local/share"))
    cfg["glossary"] = str(Path(cfg["glossary"]).expanduser() if cfg["glossary"] else path.with_name("glossary.txt"))
    v = cfg["speakers"]["voices"]
    cfg["speakers"]["voices"] = str(Path(v).expanduser() if v else path.with_name("voices"))
    cfg["sources"]["folder"]["path"] = str(Path(cfg["sources"]["folder"]["path"]).expanduser())
    if cfg["outputs"].get("markdown"):
        cfg["outputs"]["markdown"]["dir"] = str(Path(cfg["outputs"]["markdown"]["dir"]).expanduser())
    return cfg


def _refuse_secrets(d: dict, prefix: str = "") -> None:
    """A secret typed into plaudit.yaml would be silently ignored (or, for a webhook URL, backed up with the notes),
    so the file is rejected instead and the message says where the value belongs."""
    for k, v in d.items():
        hook_url = k == "url" and isinstance(v, str) and ("hooks.slack.com/" in v or "discord.com/api/webhooks/" in v)
        if k in ("api_key", "token", "password") or hook_url:
            raise ValueError(f"{prefix}{k}: secrets never go in plaudit.yaml. Set {prefix}{k}_env to a variable name "
                             f"and put the value in the environment or in the .env file next to plaudit.yaml")
        if isinstance(v, dict):
            _refuse_secrets(v, f"{prefix}{k}.")


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

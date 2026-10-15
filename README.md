# Plaudit

Plaudit turns meeting recordings from any device into speaker-labelled transcripts and summaries, on hardware you control. Chinese and English, including code-switching inside one sentence, are first-class.

Pipeline: source → 16 kHz WAV → speaker turns (Nemotron-3-Diarization) → transcription per turn (Qwen3-ASR-1.7B, glossary as context) → summary via any OpenAI-compatible LLM → Markdown folder and/or a Notion page per meeting → verified rclone backup.

Plaudit is not affiliated with Plaud. It reads Plaud recordings only through the official `@plaud-ai/cli`.

## Quickstart

```bash
pip install plaudit            # core (watched folder, API transcription, Markdown/Notion output)
pip install "plaudit[gpu]"     # adds transformers and torch for local models (Qwen3-ASR and Nemotron both ship in transformers)
sudo apt install ffmpeg        # plus rclone if you want backups, Node 22+ and `npm i -g @plaud-ai/cli` for Plaud
```

Copy `examples/plaudit.gpu.yaml` (local models) or `examples/plaudit.api.yaml` (no GPU, transcription over HTTP) to `~/plaudit/plaudit.yaml` and edit it. Secrets are not in the config file; they are environment variables, read from `~/.config/plaudit.env` (`KEY=value` lines, `chmod 600`) by the systemd unit, or exported by you before `plaudit run`. The variable names are configurable (`api_key_env` / `token_env`).

| Variable | Used for |
| --- | --- |
| `PLAUDIT_SUMMARY_API_KEY` | the chat endpoint that writes the summary (name is configurable) |
| `NOTION_TOKEN` | Notion internal integration, if the Notion output is on |
| `PLAUDIT_ASR_API_KEY` | the `/v1/audio/transcriptions` endpoint, API mode only |

Then:

```bash
plaudit process meeting.m4a          # one file, prints the summary
plaudit run                          # pull new recordings, process the queue once, exit
plaudit run --loop 300               # same, every 5 minutes; models are reloaded each cycle so memory is free in between
plaudit status                       # what is waiting, done, stuck
```

Drop a file into the watched folder (`sources.folder.path`) and it is picked up on the next run. For Plaud, run `plaud login` once, set `sources.plaud.enabled: true`, and new recordings are fetched every run.

## How it behaves

- One job at a time, behind a lock file. A run starts a job only when `MemAvailable` is above `min_free_mem_gb`, so a shared GPU box is never pushed over the edge.
- Audio is transcribed per speaker turn, max 30 s each, never as one long tensor.
- Failures stay queued and retry on the next run. After 24 h of failures a job is marked `stuck` in `plaudit status`. There is no paid fallback. A Notion page is created once; if appending the transcript fails, the page stays without it rather than being duplicated on retry.
- A backup that fails is retried on every later run until it passes.
- Local copies are deleted only after `rclone check` has verified the backup and `retention_days` have passed.
- The glossary (`glossary.txt`, one term per line) is read fresh for every job and passed to the ASR model and the summariser.

## Plug-in points

| Stage | Options in v0.1 |
| --- | --- |
| Source | watched folder, Plaud (official CLI, timed check) |
| Speakers | `nemotron`, `none` |
| Transcription | `qwen_asr` (local), `openai` (any `/v1/audio/transcriptions`) |
| Summary | any OpenAI-compatible `/v1/chat/completions` |
| Output | Markdown folder, Notion database |
| Backup | any rclone remote |

Each is a small class with one method; see `src/plaudit/`. `plaudit notion-init <parent page id>` creates the Notion database with the properties Plaudit writes (`Name`, `Date`, `Duration (min)`, `Source`, `Status`) and prints its id for `plaudit.yaml`. A job that keeps failing for 24 h gets a stub page with status `stuck`.

## Docker

`Dockerfile` builds an image with ffmpeg, rclone, Node 22, the Plaud CLI and the GPU extras. The default base image is for aarch64 (DGX Spark); pass `--build-arg BASE=vllm/vllm-openai:v0.29.0` on x86_64, or `--build-arg BASE=python:3.12-slim --build-arg EXTRAS=` for the CPU/API mode. `docker-compose.yml` and `deploy/systemd/` show a per-job container started by a systemd user timer with a hard memory cap; mount `/etc/localtime` so meeting dates are in your time zone.

## License

Apache-2.0.

# Plaudit

Plaudit turns meeting recordings from any device into speaker-labelled transcripts and summaries, on hardware you control. Chinese and English, including code-switching inside one sentence, are first-class.

Pipeline: source → 16 kHz WAV → speaker turns (Nemotron-3-Diarization) → transcription per turn (Qwen3-ASR-1.7B, glossary as context) → minutes via any OpenAI-compatible LLM → Markdown file and/or a Notion page per meeting → verified rclone backup.

Plaudit is not affiliated with Plaud. It reads Plaud recordings only through the official `@plaud-ai/cli`, and never writes to or deletes from your Plaud account.

## Install

```bash
pip install "plaudit @ git+https://github.com/Tianyi-Billy-Ma/Plaudit"          # core: watched folder, transcription over HTTP, Markdown/Notion output
pip install "plaudit[gpu] @ git+https://github.com/Tianyi-Billy-Ma/Plaudit"     # adds transformers, torch, numpy, librosa for the local models
sudo apt install ffmpeg          # required (ffmpeg and ffprobe)
sudo apt install rclone          # only for backups
npm i -g @plaud-ai/cli           # only for the Plaud source (Node 22+), then `plaud login` once
```

Python 3.10+. Local models need transformers 5.19+ and a CUDA GPU with roughly 5 GB free for the two models.

## Configure

Copy `examples/plaudit.gpu.yaml` (local models) or `examples/plaudit.api.yaml` (no GPU, transcription over HTTP) to `~/plaudit/plaudit.yaml` and edit it. Another location works with `plaudit -c path/plaudit.yaml` or `PLAUDIT_CONFIG=path/plaudit.yaml`. Every key is optional; the defaults are in `src/plaudit/config.py`.

| Key | Default | Meaning |
| --- | --- | --- |
| `data_dir` | `~/plaudit/data` | queue state: `inbox/` (downloaded audio), `jobs/` (one JSON per job), `done/` (one folder per finished job), `seen.json` (Plaud ids already handled), `plaudit.lock` |
| `glossary` | `~/plaudit/glossary.txt` | one term per line, `#` comments; read fresh for every job and given to the ASR model and the summariser |
| `language` | `null` | ASR language hint: `null` auto-detects, or `Chinese` / `English` |
| `sources.folder.path` | `~/plaudit/inbox` | watched folder; audio files (`.mp3 .wav .m4a .aac .ogg .opus .flac .mp4 .webm .wma`) are moved into the queue once they are at least 10 s old |
| `sources.plaud.enabled` | `false` | fetch new Plaud recordings on every run (needs `plaud login`) |
| `sources.plaud.lookback_days` | `7` | ignore recordings uploaded before this; clips under 60 s are ignored too, and both are remembered in `seen.json` so they are never re-checked |
| `diarization.backend` | `none` | `nemotron` (local model) or `none` (one speaker, still chunked) |
| `diarization.model` | `nvidia/Nemotron-3-Diarization` | Hugging Face id |
| `diarization.max_turn_seconds` | `30` | longer speaker turns are split before transcription |
| `transcription.backend` | `qwen_asr` | `qwen_asr` (local, through transformers) or `openai` (any `/v1/audio/transcriptions` endpoint) |
| `transcription.model` | `Qwen/Qwen3-ASR-1.7B-hf` | Hugging Face id, or the model name the HTTP endpoint expects |
| `transcription.base_url` | `null` | `openai` backend only, e.g. `http://localhost:8080/v1` |
| `transcription.api_key_env` | unset | `openai` backend only: name of the environment variable holding the bearer token |
| `summary.base_url` | `null` | `/v1` of any OpenAI-compatible chat endpoint; `null` disables the summary (the notes then hold the transcript only) |
| `summary.model` | `glm-5.3-flash` | model name sent in the request |
| `summary.api_key_env` | `OPENAI_API_KEY` | environment variable holding the bearer token |
| `summary.language` | `zh` | language of the minutes: `zh` (摘要 / 决定 / 行动项, English terms kept) or `en` (Summary / Decisions / Action items) |
| `summary.chunk_chars` | `12000` | transcripts longer than this are summarised in parts and the parts merged; no single request is longer than this |
| `summary.reasoning_effort` | unset | sent as-is in the request; `none` switches thinking off for reasoning models (GLM, Qwen3, DeepSeek), which otherwise think for minutes per part and may return nothing. Leave unset for models that reject the field |
| `outputs.markdown.dir` | `~/plaudit/data/notes` | one `YYYY-MM-DD <title>.md` per meeting; `null` turns Markdown output off |
| `outputs.notion.database_id` | `null` | set to publish one page per meeting into an existing database |
| `outputs.notion.token_env` | `NOTION_TOKEN` | environment variable holding the Notion internal-integration token |
| `outputs.notion.status_done` / `status_stuck` | `done` / `stuck` | option names to write into the database's `Status` column |
| `backup.remote` | `null` | rclone remote and folder, e.g. `gdrive:Meetings`; finished jobs are copied to `YYYY/MM/<job id>/` and verified with `rclone check` |
| `retention_days` | `7` | delete the local copy of a job this long after it finished, and only if its backup was verified |
| `min_free_mem_gb` | `8` | do not load the models while `MemAvailable` is below this |

### Secrets

Secrets never go in `plaudit.yaml`. Plaudit reads them from environment variables, and fills in any that are missing from a `.env` file next to `plaudit.yaml` (`KEY=value` lines, `export` prefix and quotes allowed, `chmod 600`). The variable names are the `api_key_env` / `token_env` values above:

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | the chat endpoint that writes the minutes |
| `NOTION_TOKEN` | Notion internal integration, when `outputs.notion.database_id` is set |
| `PLAUDIT_ASR_API_KEY` | the transcription endpoint, as named in `examples/plaudit.api.yaml` |

The Plaud CLI keeps its own login in `~/.plaud`, rclone its own in `~/.config/rclone/rclone.conf`.

## Run

```bash
plaudit run                  # pull new recordings, process the queue once, back up, exit
plaudit run --loop 300       # the same every 5 minutes; models are reloaded each cycle so memory is free in between
plaudit pull                 # only fetch new recordings into the queue
plaudit status               # one line per job: id, waiting / delayed / running / done / stuck / retired, title, last error
plaudit process meeting.m4a [--title T] [--date YYYY-MM-DD]   # one file now; prints the minutes and the output links
plaudit notion-init <parent page id> [--title Meetings]       # create a Notion database for Plaudit, prints its id
```

Drop a file into the watched folder and it is queued on the next run. With the Plaud source on, new recordings are fetched every run; Plaud has no webhook, so a timer is the only way. `plaudit process` takes the same lock as `run`, writes its result into the queue as a finished job (so it is backed up and retired like any other) and does not touch the file you gave it.

## How it behaves

- **One job at a time.** A run holds `plaudit.lock` in `data_dir`; a second run started meanwhile logs that and exits. Models are loaded only when `MemAvailable` is above `min_free_mem_gb`, so a box that also serves an LLM is never pushed over the edge. Once loaded, they stay for the rest of that run.
- **Per speaker turn.** Audio is transcribed turn by turn, at most `max_turn_seconds` each, never as one long tensor. Diarization streams the recording in fixed chunks, so memory stays flat for multi-hour files.
- **Retries, with backoff.** A failed job goes back to `waiting` and is retried after 10 min, then 20, 40, up to 2 h between attempts. The transcript is saved as `transcript.json` after diarization and ASR, so a retry after a summary or output failure skips straight to the summary. After 24 h of failures the job is marked `stuck`; if the Notion output is on, a stub page with status `stuck` and the error is created. A run killed mid-job (out of memory, reboot) counts as a failed attempt, and a job whose third attempt ends that way is marked `stuck` at once. There is no paid fallback.
- **Summary of long meetings.** The transcript is split into parts of `chunk_chars`, each part is summarised, and the part minutes are merged in rounds until one set remains. Every answer is cached in `summary-cache.json` in the job folder (key: model + prompts), so a dropped connection or a retry does not repeat the calls already made. If the model returns no text (a reasoning model that spent its budget thinking, or a reply cut off after minutes), the part is halved and tried again; set `summary.reasoning_effort: none` for such models.
- **Outputs are written once.** The Markdown file is never overwritten; a second meeting with the same title on the same day gets the job id appended. A Notion page is created once (the create call is not retried); if appending the transcript fails, the page keeps its minutes rather than being duplicated on the next attempt.
- **Backup, then retention.** Every finished job without a verified backup is backed up on every later run until it passes. Local copies (converted audio removed, original audio, `notes.md`, `transcript.json`) are deleted only after a verified backup and `retention_days`.
- **Dates are local.** Plaud reports times in UTC; Plaudit converts them to the machine's time zone (mount `/etc/localtime` into a container).

## Notion

`plaudit notion-init <parent page id>` creates a database with the columns Plaudit writes: `Name` (title), `Date` (date), `Duration (min)` (number), `Source` (text), `Status` (select: `done`, `stuck`). It prints the database id for `outputs.notion.database_id`; share the parent page with your integration first.

An existing database works too. Columns are matched by name and type and missing ones are skipped: `Date` may be a date or text column, `Source` text (`<source> <id>`) or a select set to `Plaudit`, `Status` a select or status column (names from `status_done` / `status_stuck`). If the database has `Page Type` (multi-select) or `PARA` (select) columns they are set to `Meeting`, `Audio` and `Resource`. The minutes become the page body (headings, bullets, numbered items), the transcript goes into a `Transcript` toggle, 100 blocks per request and 2000 characters per line.

## Docker and systemd

`Dockerfile` builds an image with ffmpeg, rclone, Node 22, the Plaud CLI and the `gpu` extras. The default base image is the aarch64 vLLM image (DGX Spark / GB10); on x86_64 pass `--build-arg BASE=vllm/vllm-openai:v0.29.0`, and for the CPU/API mode `--build-arg BASE=python:3.12-slim --build-arg EXTRAS=`. The image reads `/plaudit/plaudit.yaml` and expects these mounts: `/plaudit` (config, glossary, data), `/root/.cache/huggingface` (models), `/root/.plaud` (Plaud login), `/root/.config/rclone` (read-only), `/etc/localtime` (read-only).

`docker-compose.yml` runs one container per `docker compose run --rm plaudit`; `deploy/systemd/plaudit.service` and `plaudit.timer` start one such container every 5 minutes as a systemd user unit, with a hard memory cap and an 8 h limit per run (a 2 h recording with a slow summariser needs about 3.5 h). The unit sources `~/.config/secrets.env` in a shell and passes `OPENAI_API_KEY` and `NOTION_TOKEN` into the container; a `.env` next to `plaudit.yaml` inside the mounted folder works as well. Copy the two unit files to `~/.config/systemd/user/` and run `systemctl --user enable --now plaudit.timer`; stop everything with `systemctl --user disable --now plaudit.timer`.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests use fake diarization, ASR and summary backends and need only ffmpeg. The two local-model backends (`NemotronDiarizer`, `QwenASRTranscriber`) have no automated test; they are exercised on the reference GPU box. Each stage is a small class with one method (`diarize`, `transcribe`, `summarize`, `publish`), registered in the `BACKENDS` dict of its module.

## License

Apache-2.0.

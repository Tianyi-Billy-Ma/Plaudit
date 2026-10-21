from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from importlib import resources
from datetime import date, datetime
from pathlib import Path

from . import __version__, config
from .daemon import Queue, due
from .pipeline import Pipeline
from .sources import new_job

log = logging.getLogger(__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="plaudit", description="Meeting recordings to transcripts and summaries.")
    ap.add_argument("--config", "-c", help="path to plaudit.yaml (default ~/plaudit/plaudit.yaml or $PLAUDIT_CONFIG)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="pull sources and process the queue once (use --loop to keep going)")
    run.add_argument("--loop", type=int, metavar="SECONDS", help="repeat every N seconds instead of exiting")
    one = sub.add_parser("process", help="process one audio file now, outside the queue")
    one.add_argument("file")
    one.add_argument("--title")
    one.add_argument("--date", default=date.today().isoformat())
    sub.add_parser("pull", help="only fetch new recordings into the queue")
    sv = sub.add_parser("serve", help="HTTP upload endpoint into the watched folder (token from $PLAUDIT_UPLOAD_TOKEN)")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, help="default sources.upload.port (8787)")
    sub.add_parser("status", help="list queued jobs")
    init = sub.add_parser("init", help="write a starter plaudit.yaml, inbox/ and glossary.txt (never overwrites)")
    init.add_argument("--api", action="store_true", help="the no-GPU config (transcription and summary over HTTP)")
    ni = sub.add_parser("notion-init", help="create the Meetings database under a Notion page (token from $NOTION_TOKEN)")
    ni.add_argument("parent_page_id")
    ni.add_argument("--title", default="Meetings")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "init":
        return init_config(config.config_path(args.config), "api" if args.api else "gpu")
    cfg = config.load(args.config)
    if args.cmd == "notion-init":
        from .outputs import create_notion_database

        token = config.secret(cfg["outputs"]["notion"]["token_env"])
        if not token:
            ap.error(f"set ${cfg['outputs']['notion']['token_env']} first")
        print(create_notion_database(token, args.parent_page_id, args.title))
        return 0
    if args.cmd == "serve":
        from .upload import serve

        serve(cfg, args.host, args.port)
        return 0
    q = Queue(cfg)
    if args.cmd == "run":
        while True:
            if not args.loop:
                q.run_once()
                return 0
            try:
                q.run_once()
            except Exception:
                log.exception("run failed; next cycle in %ss", args.loop)
            time.sleep(args.loop)
    if args.cmd == "pull":
        print(f"queued {q.pull()} new recording(s)")
        return 0
    if args.cmd == "status":
        for j in q.jobs():
            status = j["status"] if due(j) or j["status"] != "waiting" else "delayed"
            print(f"{j['id']}  {status:8}  {j['title']}  {j.get('last_error', '')}")
        return 0
    src = Path(args.file)
    job = new_job(src, args.title or src.stem, "manual", when=args.date)
    with q.lock() as got:
        if not got:
            ap.error("another plaudit run is in progress; try again in a minute")
        workdir = q.done_dir / job["id"]
        result = Pipeline(cfg).process(src, job, workdir)
        shutil.copy(src, workdir)  # the job is saved as done, so retire() backs it up and deletes it like any other
        job.update(status="done", links=result["links"], duration_min=result["meta"]["duration_min"],
                   finished_at=datetime.now().isoformat(timespec="seconds"))
        q.save(job)
        q.retire()
    print(result["summary"] or result["transcript"].as_text())
    print("\n".join(result["links"]))
    return 0


def init_config(path: Path, flavour: str) -> int:
    if path.exists():
        print(f"{path} exists, leaving it alone")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resources.files("plaudit.examples").joinpath(f"plaudit.{flavour}.yaml").read_text())
        print(f"wrote {path}")
    cfg = config.load(path)
    Path(cfg["sources"]["folder"]["path"]).expanduser().mkdir(parents=True, exist_ok=True)
    glossary = Path(cfg["glossary"]).expanduser()
    if not glossary.exists():
        glossary.write_text("# one term per line, given to the ASR model and the summariser\n")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"warning: {tool} not found on PATH; install ffmpeg before processing audio")
    print(f"edit {path}, then: plaudit process meeting.m4a  (or drop files into {cfg['sources']['folder']['path']} and run: plaudit run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

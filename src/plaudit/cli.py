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
    ap.add_argument("--config", "-c", help="path to plaudit.yaml (default ~/.config/plaudit/plaudit.yaml or $PLAUDIT_CONFIG)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="process one recording (run INPUT [OUTPUT]), or without INPUT pull the sources and "
                                     "process the queue once (--loop to keep going)")
    run.add_argument("input", nargs="?", metavar="INPUT", help="an audio file to process now, outside the queue")
    run.add_argument("output", nargs="?", metavar="OUTPUT",
                     help="where its notes go: a .md file or a folder (default outputs.markdown.dir)")
    run.add_argument("--title", help="meeting title (default: the file name)")
    run.add_argument("--date", default=date.today().isoformat(), help="meeting date YYYY-MM-DD (default: today)")
    run.add_argument("--loop", type=int, metavar="SECONDS", help="queue mode: repeat every N seconds instead of exiting")
    sub.add_parser("pull", help="only fetch new recordings into the queue")
    sv = sub.add_parser("serve", help="HTTP upload endpoint into the watched folder (token from $PLAUDIT_UPLOAD_TOKEN)")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, help="default sources.upload.port (8787)")
    sub.add_parser("status", help="list queued jobs")
    rn = sub.add_parser("rename", help="name the speakers of a finished job: plaudit rename <job id> 'Speaker 2=张三' ...")
    rn.add_argument("job", help="job id as shown by `plaudit status`")
    rn.add_argument("names", nargs="+", metavar="LABEL=NAME")
    rn.add_argument("--enrol", action="store_true", help="also save their voice in speakers.voices, so the next meeting names them")
    vo = sub.add_parser("voices", help="list the people Plaudit can recognise, or `voices add NAME FILE` to add a recording of one")
    vo.add_argument("add", nargs="?", choices=["add"])
    vo.add_argument("name", nargs="?", metavar="NAME")
    vo.add_argument("file", nargs="?", metavar="FILE", help="a recording of NAME alone (any audio format; 20-30 s is enough)")
    init = sub.add_parser("init", help="write a starter plaudit.yaml and glossary.txt into ~/.config/plaudit (never overwrites)")
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
    if args.cmd == "voices":
        from .audio import duration_seconds
        from .voices import add_sample, samples

        if args.add:
            if not (args.name and args.file and Path(args.file).is_file()):
                ap.error("usage: plaudit voices add NAME FILE")
            try:
                print(f"saved {add_sample(cfg, args.name, Path(args.file))}")
            except ValueError as e:
                ap.error(str(e))
        for name, files in samples(Path(cfg["speakers"]["voices"])).items():
            print(f"{name}  {len(files)} sample(s), {sum(duration_seconds(f) for f in files):.0f} s  ({files[0].parent})")
        return 0
    q = Queue(cfg)
    if args.cmd == "run" and not args.input:
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
    if args.cmd == "rename":
        from .voices import rename_job

        job = next((j for j in q.jobs() if j["id"] == args.job and j["status"] == "done"), None)
        if not job:
            ap.error(f"no finished job {args.job!r} (see `plaudit status`; retired jobs are gone from this machine)")
        names = dict(n.split("=", 1) for n in args.names if "=" in n)
        if len(names) != len(args.names):
            ap.error("each name must look like 'Speaker 2=张三'")
        try:
            rename_job(cfg, q.done_dir / job["id"], names, job.get("links", []), enrol=args.enrol)
        except ValueError as e:
            ap.error(str(e))
        print(f"renamed {', '.join(f'{k} -> {v}' for k, v in names.items())} in {q.done_dir / job['id']}"
              + (" and saved their voices" if args.enrol else ""))
        return 0
    src = Path(args.input)
    if not src.is_file():
        ap.error(f"{src} is not a file")
    job = new_job(src, args.title or src.stem, "manual", when=args.date)
    with q.lock() as got:
        if not got:
            ap.error("another plaudit run is in progress; try again in a minute")
        workdir = q.done_dir / job["id"]
        sinks = None
        if args.output:  # the Markdown note goes where asked; every other configured output still runs
            from .outputs import MarkdownOutput, build

            sinks = [s for s in build(cfg) if not isinstance(s, MarkdownOutput)]
            sinks.insert(0, MarkdownOutput(cfg, Path(args.output).expanduser()))
        result = Pipeline(cfg, sinks=sinks).process(src, job, workdir)
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
    print(f"edit {path}, then: plaudit run meeting.m4a [notes.md]  (or drop files into {cfg['sources']['folder']['path']} and run: plaudit run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

from . import __version__, config
from .daemon import Queue
from .pipeline import Pipeline
from .sources import new_job


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
    sub.add_parser("status", help="list queued jobs")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load(args.config)
    q = Queue(cfg)
    if args.cmd == "run":
        while True:
            q.run_once()
            if not args.loop:
                return 0
            time.sleep(args.loop)
    if args.cmd == "pull":
        print(f"queued {q.pull()} new recording(s)")
        return 0
    if args.cmd == "status":
        for j in q.jobs():
            print(f"{j['id']}  {j['status']:8}  {j['title']}  {j.get('last_error', '')}")
        return 0
    src = Path(args.file)
    job = new_job(q.data / "inbox", src, args.title or src.stem, "manual", when=args.date)
    result = Pipeline(cfg).process(src, job, q.done_dir / job["id"])
    print(result["summary"] or result["transcript"].as_text())
    print("\n".join(result["links"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

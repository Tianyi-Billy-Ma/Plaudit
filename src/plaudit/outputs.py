"""Where results land: a Markdown folder, a Notion database (one page per meeting), a generic JSON webhook and/or
notifications through Apprise (Slack, Discord, Teams, Telegram, e-mail...). The notifying outputs run last and
receive the links the others produced."""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

try:
    import apprise  # optional: pip install plaudit[notify]
except ImportError:  # pragma: no cover
    apprise = None

from .config import secret
from .http import get_json, post_json
from .models import Transcript, _ts

log = logging.getLogger(__name__)
NOTION = "https://api.notion.com/v1"


def render_markdown(meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None) -> str:
    # YAML front matter: Obsidian and Logseq show it as note properties. json.dumps gives a valid YAML scalar.
    props = {"title": meta["title"], "date": meta["date"], "duration_min": meta["duration_min"],
             "source": f"{meta['source']} {meta.get('source_id', '')}".strip(), "audio": audio_link}
    head = ["---", *(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in props.items() if v is not None), "---", "",
            f"# {meta['title']}"]
    return "\n".join(head) + "\n\n" + summary.strip() + "\n\n## Transcript\n\n" + transcript.as_text() + "\n"


class MarkdownOutput:
    def __init__(self, cfg: dict):
        self.dir = Path(cfg["outputs"]["markdown"]["dir"])

    def publish(self, meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None, links=()) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{meta['date']} {safe(meta['title'])}.md"
        if path.exists():  # second meeting with the same title that day
            path = path.with_stem(f"{path.stem} {meta['id']}")
        path.write_text(render_markdown(meta, summary, transcript, audio_link))
        return str(path)


class NotionOutput:
    """One page per meeting in an existing database. Columns are matched by name AND type, so the database
    may be Plaudit's own (see create_notion_database) or a user's with different column types; missing
    columns are skipped. Status option names come from the config (`status_done`, `status_stuck`)."""

    def __init__(self, cfg: dict):
        nc = cfg["outputs"]["notion"]
        self.database_id = nc["database_id"]
        token = secret(nc["token_env"])
        if not (self.database_id and token):
            raise RuntimeError("Notion output needs outputs.notion.database_id and the token env var")
        self.headers = {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28"}
        self.status = {"done": nc.get("status_done", "done"), "stuck": nc.get("status_stuck", "stuck")}
        self.schema = {k: v["type"] for k, v in get_json(f"{NOTION}/databases/{self.database_id}", self.headers)["properties"].items()}

    def _props(self, meta: dict, status: str) -> dict:
        title_col = next(k for k, t in self.schema.items() if t == "title")
        props = {title_col: {"title": [{"text": {"content": meta["title"]}}]}}
        source = f"{meta['source']} {meta.get('source_id', '')}".strip()
        candidates = {
            "Date": {"date": {"start": meta["date"]}, "rich_text": [{"text": {"content": meta["date"]}}]},
            "Duration (min)": {"number": meta.get("duration_min")},
            "Source": {"rich_text": [{"text": {"content": source}}], "select": {"name": "Plaudit"}},
            "Status": {"select": {"name": self.status[status]}, "status": {"name": self.status[status]}},
            "Page Type": {"multi_select": [{"name": "Meeting"}, {"name": "Audio"}]},
            "PARA": {"select": {"name": "Resource"}},
        }
        for col, by_type in candidates.items():
            t = self.schema.get(col)
            if t in by_type and by_type[t] is not None:
                props[col] = {t: by_type[t]}
        return props

    def publish(self, meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None, links=()) -> str:
        children = md_to_blocks(summary)
        if audio_link:
            children.append(_para(f"Audio: {audio_link}", link=audio_link))
        page = post_json(f"{NOTION}/pages", retries=1, payload={"parent": {"database_id": self.database_id},
                                              "properties": self._props(meta, "done"),  # meta["status"] is the job state (waiting), not the column
                                              "children": children[:100]}, headers=self.headers)
        url = page.get("url", page["id"])
        # Transcript goes in a toggle; Notion caps 100 blocks per call and 2000 chars per rich text.
        # The page exists now: a failure below is logged, not raised, so a retry cannot create a duplicate page.
        try:
            toggle = post_json(f"{NOTION}/blocks/{page['id']}/children",
                               {"children": [{"object": "block", "type": "toggle",
                                              "toggle": {"rich_text": [{"text": {"content": "Transcript"}}]}}]},
                               self.headers, method="PATCH")  # Notion appends children with PATCH
            toggle_id = toggle["results"][0]["id"]
            lines = [_para(f"[{_ts(s.start)}] {s.speaker}: {s.text}"[:2000]) for s in transcript.segments if s.text]
            for i in range(0, len(lines), 100):
                post_json(f"{NOTION}/blocks/{toggle_id}/children", {"children": lines[i:i + 100]}, self.headers, method="PATCH")
        except Exception:
            log.exception("page %s created but the transcript could not be appended", url)
        return url

    def publish_stuck(self, meta: dict, error: str) -> str:
        """A stub page so a job that has failed for 24 h is visible where the meetings are."""
        page = post_json(f"{NOTION}/pages", retries=1, payload={"parent": {"database_id": self.database_id},
                                              "properties": self._props(meta, "stuck"),
                                              "children": [_para(f"Plaudit could not process this recording: {error}"[:2000])]},
                         headers=self.headers)
        return page.get("url", page["id"])


def headline(meta: dict, links=()) -> str:
    return f"{meta['title']} ({meta['date']}, {meta.get('duration_min', '?')} min)" + "".join(f"\n{l}" for l in links)


class WebhookOutput:
    """One POST of the whole result as JSON per finished (or stuck) meeting, for n8n, Zapier, Make, Home Assistant...
    For a chat service use outputs.notify instead."""

    def __init__(self, cfg: dict):
        wc = cfg["outputs"]["webhook"]
        self.url = secret(wc.get("url_env")) or wc.get("url")  # the URL may carry a token
        if not self.url:
            raise RuntimeError("outputs.webhook needs a url or url_env")

    def _send(self, event: dict) -> str:
        post_json(self.url, event, retries=1)  # a notification is not worth a duplicate on retry
        return "webhook"  # not the URL: links are logged, kept in the job file and backed up

    def publish(self, meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None, links=()) -> str:
        return self._send({"event": "meeting.done", "meta": meta, "summary": summary, "links": list(links),
                           "audio_link": audio_link, "transcript": transcript.to_dict()})

    def publish_stuck(self, meta: dict, error: str) -> str:
        return self._send({"event": "meeting.stuck", "meta": meta, "error": error})


class NotifyOutput:
    """The minutes to Slack, Discord, Teams, Telegram, Matrix, e-mail, ntfy... through Apprise: one URL per
    destination (https://github.com/caronc/apprise/wiki). URLs carry tokens, so they normally come from the
    environment; the link Plaudit records is Apprise's masked form."""

    def __init__(self, cfg: dict):
        if apprise is None:
            raise RuntimeError("outputs.notify needs Apprise: pip install 'plaudit[notify]'")
        nc = cfg["outputs"]["notify"]
        self.app = apprise.Apprise()
        for url in list(nc.get("urls") or []) + re.split(r"[\s,]+", secret(nc.get("urls_env")) or ""):
            if url and not self.app.add(url):
                raise RuntimeError(f"outputs.notify: Apprise rejected a {url.split(':')[0]}:// URL")
        self.label = ", ".join(self.app.urls(privacy=True))

    def _send(self, title: str, body: str, notes: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as d:  # attachments are passed by path
            attach = None
            if notes:
                attach = str(Path(d) / "notes.md")
                Path(attach).write_text(notes)
            ok = self.app.notify(title=title, body=body, body_format=apprise.NotifyFormat.MARKDOWN, attach=attach)
        if not ok:  # logged, not raised: a retry would duplicate the Markdown/Notion outputs for a lost notification
            log.error("notification failed for %s", self.label)
        return self.label

    def publish(self, meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None, links=()) -> str:
        return self._send(headline(meta), "".join(f"{l}\n" for l in links) + "\n" + summary.strip(),
                          render_markdown(meta, summary, transcript, audio_link))

    def publish_stuck(self, meta: dict, error: str) -> str:
        return self._send(f"Plaudit could not process {headline(meta)}", error)



def create_notion_database(token: str, parent_page_id: str, title: str = "Meetings") -> str:
    """Create the Meetings database with the properties NotionOutput writes. Returns its id for plaudit.yaml."""
    db = post_json(f"{NOTION}/databases", {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"text": {"content": title}}],
        "properties": {
            "Name": {"title": {}},
            "Date": {"date": {}},
            "Duration (min)": {"number": {}},
            "Source": {"rich_text": {}},
            "Status": {"select": {"options": [{"name": "done", "color": "green"}, {"name": "stuck", "color": "red"}]}},
        },
    }, {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28"})
    return db["id"]


def _para(text: str, link: str | None = None) -> dict:
    rt = {"text": {"content": text}}
    if link:
        rt["text"]["link"] = {"url": link}
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [rt]}}


def md_to_blocks(md: str) -> list[dict]:
    """Minimal Markdown to Notion blocks: ##/### headings, - bullets, 1. numbered, paragraphs."""
    blocks = []
    for line in md.splitlines():
        s = line.strip()[:2000]  # Notion rejects any rich text over 2000 chars
        if not s:
            continue
        if s.startswith("### "):
            blocks.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [{"text": {"content": s[4:]}}]}})
        elif s.startswith("## ") or s.startswith("# "):
            blocks.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": s.lstrip('# ')}}]}})
        elif s[:2] in ("- ", "* "):
            blocks.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": s[2:]}}]}})
        elif s[0].isdigit() and ". " in s[:4]:
            blocks.append({"object": "block", "type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"text": {"content": s.split(". ", 1)[1]}}]}})
        else:
            blocks.append(_para(s))
    return blocks


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_.()[]" or ord(c) > 127 else "_" for c in name).strip()[:80] or "meeting"


def build(cfg: dict) -> list:
    outs = []
    if cfg["outputs"].get("markdown"):
        outs.append(MarkdownOutput(cfg))
    if cfg["outputs"].get("notion", {}).get("database_id"):
        outs.append(NotionOutput(cfg))
    if cfg["outputs"].get("webhook", {}).get("url") or secret(cfg["outputs"].get("webhook", {}).get("url_env")):
        outs.append(WebhookOutput(cfg))
    if cfg["outputs"].get("notify", {}).get("urls") or secret(cfg["outputs"].get("notify", {}).get("urls_env")):
        outs.append(NotifyOutput(cfg))
    return outs

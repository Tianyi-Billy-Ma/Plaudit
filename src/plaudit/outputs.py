"""Where results land: a Markdown folder and/or a Notion database (one page per meeting)."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import secret
from .http import get_json, post_json
from .models import Transcript

log = logging.getLogger(__name__)
NOTION = "https://api.notion.com/v1"


def render_markdown(meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None) -> str:
    head = [f"# {meta['title']}", "", f"- Date: {meta['date']}", f"- Duration: {meta['duration_min']} min",
            f"- Source: {meta['source']} {meta.get('source_id', '')}".rstrip()]
    if audio_link:
        head.append(f"- Audio: {audio_link}")
    return "\n".join(head) + "\n\n" + summary.strip() + "\n\n## Transcript\n\n" + transcript.as_text() + "\n"


class MarkdownOutput:
    def __init__(self, cfg: dict):
        self.dir = Path(cfg["outputs"]["markdown"]["dir"])

    def publish(self, meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{meta['date']} {safe(meta['title'])}.md"
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
            "Page Type": {"multi_select": [{"name": "Meeting"}]},
        }
        for col, by_type in candidates.items():
            t = self.schema.get(col)
            if t in by_type and by_type[t] is not None:
                props[col] = {t: by_type[t]}
        return props

    def publish(self, meta: dict, summary: str, transcript: Transcript, audio_link: str | None = None) -> str:
        children = md_to_blocks(summary)
        if audio_link:
            children.append(_para(f"Audio: {audio_link}", link=audio_link))
        page = post_json(f"{NOTION}/pages", {"parent": {"database_id": self.database_id},
                                              "properties": self._props(meta, meta.get("status", "done")),
                                              "children": children[:100]}, self.headers)
        # Transcript goes in a toggle; Notion caps 100 blocks per call and 2000 chars per rich text.
        toggle = post_json(f"{NOTION}/blocks/{page['id']}/children",
                           {"children": [{"object": "block", "type": "toggle",
                                          "toggle": {"rich_text": [{"text": {"content": "Transcript"}}]}}]},
                           self.headers, method="PATCH")  # Notion appends children with PATCH
        toggle_id = toggle["results"][0]["id"]
        lines = [_para(f"[{_ts(s.start)}] {s.speaker}: {s.text}"[:2000]) for s in transcript.segments if s.text]
        for i in range(0, len(lines), 100):
            post_json(f"{NOTION}/blocks/{toggle_id}/children", {"children": lines[i:i + 100]}, self.headers, method="PATCH")
        return page.get("url", page["id"])

    def publish_stuck(self, meta: dict, error: str) -> str:
        """A stub page so a job that has failed for 24 h is visible where the meetings are."""
        page = post_json(f"{NOTION}/pages", {"parent": {"database_id": self.database_id},
                                              "properties": self._props(meta, "stuck"),
                                              "children": [_para(f"Plaudit could not process this recording: {error}"[:2000])]},
                         self.headers)
        return page.get("url", page["id"])


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


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _para(text: str, link: str | None = None) -> dict:
    rt = {"text": {"content": text}}
    if link:
        rt["text"]["link"] = {"url": link}
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [rt]}}


def md_to_blocks(md: str) -> list[dict]:
    """Minimal Markdown to Notion blocks: ##/### headings, - bullets, 1. numbered, paragraphs."""
    blocks = []
    for line in md.splitlines():
        s = line.strip()
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
            blocks.append(_para(s[:2000]))
    return blocks


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_.()[]" or ord(c) > 127 else "_" for c in name).strip()[:80] or "meeting"


def build(cfg: dict) -> list:
    outs = []
    if cfg["outputs"].get("markdown"):
        outs.append(MarkdownOutput(cfg))
    if cfg["outputs"].get("notion", {}).get("database_id"):
        outs.append(NotionOutput(cfg))
    return outs

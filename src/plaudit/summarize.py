"""Meeting summary through any OpenAI-compatible chat endpoint. Long transcripts: chunk, then merge."""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
from pathlib import Path

from .config import secret
from .http import post_json

log = logging.getLogger(__name__)

PROMPTS = {
    "zh": (
        "你是会议记录助手。根据下面带说话人标签的会议转录写出会议纪要。"
        "用中文书写，英文术语、人名、项目名保持原文。只输出以下三个部分，用 Markdown 二级标题：\n"
        "## 摘要\n（3 到 8 条要点）\n## 决定\n（已经做出的决定；没有则写“无”）\n"
        "## 行动项\n（每条：负责人（如有）、事项、截止日期（如提到））\n"
    ),
    "en": (
        "You are a meeting-notes assistant. From the speaker-labelled transcript below write the minutes. "
        "Keep names, project names and technical terms as spoken. Output exactly these Markdown sections:\n"
        "## Summary\n(3 to 8 bullet points)\n## Decisions\n(decisions made; write 'None' if none)\n"
        "## Action items\n(each: owner if known, task, due date if mentioned)\n"
    ),
}
MERGE = {
    "zh": "下面是同一场会议各部分的纪要。请合并为一份完整纪要，去重，保留同样的三个部分（摘要、决定、行动项），总长度不超过 %d 字。",
    "en": "Below are minutes for consecutive parts of one meeting. Merge them into one set of minutes with the same three sections, removing duplicates, in at most %d characters.",
}


class EmptySummary(RuntimeError):
    """The model returned no content (a reasoning model spent its whole budget thinking)."""


class Summarizer:
    def __init__(self, cfg: dict):
        sc = cfg["summary"]
        self.url = sc["base_url"].rstrip("/") + "/chat/completions"
        self.model = sc["model"]
        self.lang = sc["language"]
        self.chunk_chars = sc["chunk_chars"]
        self.cache, self.cache_file = {}, None
        self.headers = {}
        if key := secret(sc.get("api_key_env")):
            self.headers["Authorization"] = f"Bearer {key}"

    def _chat(self, system: str, user: str) -> str:
        # Every answer is cached on disk per prompt: a 2 h recording is ~40 calls over 2.5 h, and one cut
        # connection must not throw them all away.
        key = hashlib.sha256((self.model + system + user).encode()).hexdigest()[:16]
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = out = self._chat_uncached(system, user)
        if self.cache_file:
            self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False))
        return out

    def _chat_uncached(self, system: str, user: str) -> str:
        try:
            resp = post_json(self.url, {
                "model": self.model, "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }, self.headers, retries=1)
        except http.client.IncompleteRead as e:
            # a proxy cuts replies after ~10 min, which is how long GLM thinks before giving up: same cure as empty
            raise EmptySummary(f"reply from {self.model} cut off ({e})") from None
        msg = resp["choices"][0]["message"]
        # reasoning models (GLM) put thinking in reasoning_content; content is null when the budget ran out
        if not msg.get("content"):
            raise EmptySummary(f"empty summary from {self.model} (finish_reason={resp['choices'][0].get('finish_reason')})")
        return msg["content"].strip()

    def _chat_or_split(self, system: str, text: str, prefix: str = "") -> str:
        """One call; if the model returns nothing, halve the text and merge the two answers (thinking length
        grows much faster than the prompt, so halves succeed where the whole did not)."""
        try:
            return self._chat(system, prefix + text)
        except EmptySummary:
            halves = chunk(text, len(text) // 2 + 1)
            if len(text) < 1000 or len(halves) < 2:
                raise
            log.warning("summary: no answer for %d chars, retrying as two halves", len(text))
            subs = [self._chat_or_split(system, h, prefix) for h in halves]
            try:
                return self._chat(system, self.merge_prompt + "\n\n---\n\n".join(subs))
            except EmptySummary:  # the halves' answers are still minutes; a later merge round may shrink them
                log.warning("summary: no answer merging %d halves, keeping them unmerged", len(subs))
                return "\n\n".join(subs)

    @property
    def merge_prompt(self) -> str:
        # merged minutes must stay well under chunk_chars, or merge rounds never get smaller
        return MERGE[self.lang] % (self.chunk_chars // 3) + "\n\n"

    def summarize(self, transcript_text: str, glossary: list[str] = (), cache_file: Path | None = None) -> str:
        self.cache_file = cache_file
        self.cache = json.loads(cache_file.read_text()) if cache_file and cache_file.exists() else {}
        system = PROMPTS[self.lang]
        if glossary:
            system += "\n术语表 / Glossary: " + ", ".join(glossary)
        parts = chunk(transcript_text, self.chunk_chars)
        summaries = []
        for i, p in enumerate(parts, 1):
            log.info("summary: part %d/%d (%d chars)", i, len(parts), len(p))
            summaries.append(self._chat_or_split(system, p))
        # Merge in rounds of at most chunk_chars each: reasoning models think longer than their token budget on
        # big prompts (GLM hit finish_reason=length with no answer on 12k chars), so no call may exceed one chunk.
        while len(summaries) > 1:
            groups = chunk("\n\n---\n\n".join(summaries), self.chunk_chars)
            if len(groups) >= len(summaries):  # summaries themselves too big to group; one final call, as before
                groups = ["\n\n---\n\n".join(summaries)]
            log.info("summary: merging %d parts in %d call(s)", len(summaries), len(groups))
            summaries = [self._chat_or_split(system, g, self.merge_prompt) for g in groups]
        return summaries[0]


def chunk(text: str, max_chars: int) -> list[str]:
    lines, parts, cur = text.splitlines(), [], ""
    for line in lines:
        if cur and len(cur) + len(line) + 1 > max_chars:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts or [""]

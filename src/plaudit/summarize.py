"""Meeting summary through any OpenAI-compatible chat endpoint. Long transcripts: chunk, then merge."""
from __future__ import annotations

from .config import secret
from .http import post_json

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
    "zh": "下面是同一场会议各部分的纪要。请合并为一份完整纪要，去重，保留同样的三个部分（摘要、决定、行动项）。",
    "en": "Below are minutes for consecutive parts of one meeting. Merge them into one set of minutes with the same three sections, removing duplicates.",
}


class Summarizer:
    def __init__(self, cfg: dict):
        sc = cfg["summary"]
        self.url = sc["base_url"].rstrip("/") + "/chat/completions"
        self.model = sc["model"]
        self.lang = sc["language"]
        self.chunk_chars = sc["chunk_chars"]
        self.headers = {}
        if key := secret(sc.get("api_key_env")):
            self.headers["Authorization"] = f"Bearer {key}"

    def _chat(self, system: str, user: str) -> str:
        resp = post_json(self.url, {
            "model": self.model, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, self.headers)
        msg = resp["choices"][0]["message"]
        # reasoning models (GLM) put thinking in reasoning_content; content is null when the budget ran out
        if not msg.get("content"):
            raise RuntimeError(f"empty summary from {self.model} (finish_reason={resp['choices'][0].get('finish_reason')})")
        return msg["content"].strip()

    def summarize(self, transcript_text: str, glossary: list[str] = ()) -> str:
        system = PROMPTS[self.lang]
        if glossary:
            system += "\n术语表 / Glossary: " + ", ".join(glossary)
        summaries = [self._chat(system, p) for p in chunk(transcript_text, self.chunk_chars)]
        # Merge in rounds of at most chunk_chars each: reasoning models think longer than their token budget on
        # big prompts (GLM hit finish_reason=length with no answer on 12k chars), so no call may exceed one chunk.
        while len(summaries) > 1:
            groups = chunk("\n\n---\n\n".join(summaries), self.chunk_chars)
            if len(groups) >= len(summaries):  # summaries themselves too big to group; one final call, as before
                groups = ["\n\n---\n\n".join(summaries)]
            summaries = [self._chat(system, MERGE[self.lang] + "\n\n" + g) for g in groups]
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

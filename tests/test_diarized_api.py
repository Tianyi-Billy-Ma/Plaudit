"""openai_diarized backend against a mock /v1/audio/transcriptions server (async, sync fallback, errors, splitting, resume)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import FakeSummarizer


class Mock:
    def __init__(self, mode="async", reject=False):
        self.mode, self.reject = mode, reject
        self.posts, self.polls, self.jobs = [], 0, {}
        self.ack, self.plain = {}, False


def answer(n):
    return {"text": "...", "segments": [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_01", "text": f"你好 part{n}"},
        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_00", "text": "hello"},
        {"start": 3.0, "end": 3.2, "speaker": "SPEAKER_01", "text": " "},  # empty text is dropped
    ]}


@pytest.fixture
def server():
    state = Mock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.posts.append({"body": body, "prefer": self.headers.get("Prefer"), "auth": self.headers.get("Authorization")})
            if state.reject:
                return self.reply(400, {"error": {"message": "response_format 'diarized_json' is not supported"}})
            res = {"text": "plain"} if state.plain else answer(len(state.posts))
            if state.mode in ("async", "fail") and self.headers.get("Prefer") == "respond-async":
                job = f"tr_{len(state.posts)}"
                state.jobs[job] = res
                return self.reply(202, {"id": job, "status": "queued", **state.ack})
            self.reply(200, res)  # sync server, or one that ignores Prefer

        def do_GET(self):
            job = self.path.rsplit("/", 1)[-1]
            state.polls += 1
            if job not in state.jobs:
                return self.reply(404, {"error": "no such job"})
            if state.mode == "fail":
                return self.reply(200, {"id": job, "status": "failed", "error": "GPU fell over"})
            if state.polls % 2:  # every other poll still running
                return self.reply(200, {"id": job, "status": "in_progress"})
            self.reply(200, {"id": job, "status": "completed", **state.jobs[job]})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{srv.server_port}/v1"
    yield state
    srv.shutdown()


@pytest.fixture
def api_cfg(cfg, server, monkeypatch):
    monkeypatch.setenv("ASR_KEY", "k")
    cfg["transcription"].update(backend="openai_diarized", base_url=server.url, model="SPARK-ASR",
                                api_key_env="ASR_KEY", poll_seconds=0.01)
    return cfg


def wav16k(clip, tmp_path):
    from plaudit import audio
    return audio.to_wav16k(clip, tmp_path / "a.wav")


def test_async_upload_is_one_m4a_and_polls_until_done(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    segs = transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))
    assert len(server.posts) == 1 and server.posts[0]["prefer"] == "respond-async" and server.posts[0]["auth"] == "Bearer k"
    body = server.posts[0]["body"]
    assert b'name="response_format"\r\n\r\ndiarized_json' in body and b"Content-Type: audio/mp4" in body and b"ftyp" in body
    assert server.polls == 2
    assert [(s.speaker, s.text) for s in segs] == [("Speaker 1", "你好 part1"), ("Speaker 2", "hello")]


def test_server_that_ignores_prefer_answers_synchronously(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    server.mode = "sync"
    segs = transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))
    assert server.polls == 0 and len(segs) == 2
    api_cfg["transcription"]["async"] = False
    (tmp_path / "diarized-1of1.json").unlink()  # finished parts are reused otherwise
    transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))
    assert server.posts[-1]["prefer"] is None


def test_rejected_diarized_json_says_what_to_do(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    server.reject = True
    with pytest.raises(RuntimeError, match="does not support response_format=diarized_json.*nemotron"):
        transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))


def test_failed_job_raises(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    server.mode = "fail"
    with pytest.raises(RuntimeError, match="failed: GPU fell over"):
        transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))


def test_long_recording_is_split_with_offsets(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    api_cfg["transcription"]["max_segment_minutes"] = 2.5 / 60  # the 4 s clip -> 2 parts of 2 s
    segs = transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))
    assert len(server.posts) == 2
    assert [(s.start, s.speaker, s.text) for s in segs] == [
        (0.0, "Speaker 1", "你好 part1"), (1.5, "Speaker 2", "hello"),
        (2.0, "Speaker 1b", "你好 part2"), (3.5, "Speaker 2b", "hello")]


def test_retry_after_a_failed_part_reuses_the_finished_ones(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    api_cfg["transcription"]["max_segment_minutes"] = 2.5 / 60
    wav = wav16k(clip, tmp_path)
    (tmp_path / "diarized-1of2.json").write_text('{"segments": [{"start": 0, "end": 1, "speaker": "x", "text": "cached"}]}')
    segs = transcribe.build(api_cfg).diarize(wav)
    assert len(server.posts) == 1 and segs[0].text == "cached" and (tmp_path / "diarized-2of2.json").exists()


def test_queued_reply_with_null_text_is_polled_not_taken_as_empty(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    server.ack = {"text": None}
    segs = transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))
    assert server.polls == 2 and len(segs) == 2


def test_answer_without_segments_is_an_error(api_cfg, server, clip, tmp_path):
    from plaudit import transcribe

    server.mode, server.plain = "sync", True
    with pytest.raises(RuntimeError, match="without segments"):
        transcribe.build(api_cfg).diarize(wav16k(clip, tmp_path))


def test_pipeline_uses_the_api_as_diarizer_and_transcriber(api_cfg, server, clip, tmp_path):
    from plaudit.pipeline import Pipeline

    out = Pipeline(api_cfg, summarizer=FakeSummarizer(), sinks=[]).process(clip, {"title": "t", "date": "2026-11-23", "source": "folder"}, tmp_path / "job")
    assert len(server.posts) == 1
    assert "Speaker 2: hello" in out["transcript"].as_text()

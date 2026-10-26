"""End-to-end tests against a stub OpenAI-compatible server.

test_debate.py stubs `Agent.chat`, so everything below it -- how the request
is built, how a streamed response is reassembled, where usage comes from, what
a 429 body is read for -- has never actually run in a test. These do run it,
over real HTTP through the real `openai` client, against a server in a thread.

No API key and no Ollama needed, so this runs in CI alongside the rest.

    python -m unittest test_wire -v
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI

import local_debate as ld


class StubHandler(BaseHTTPRequestHandler):
    """Serves whatever the test queued, and records what it was asked."""

    def log_message(self, *args):
        pass  # keep the test output clean

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)

        status, payload = self.server.script.pop(0) if self.server.script else (200, "ok")
        if status != 200:
            self.send_error_json(status, payload)
        elif body.get("stream"):
            # Real endpoints send the usage chunk only when asked, so the stub
            # must too -- otherwise a client that forgot to ask still passes.
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            self.send_stream(payload, include_usage)
        else:
            self.send_json(payload)

    def send_error_json(self, status, message):
        raw = json.dumps({"error": {"code": status, "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_json(self, text):
        raw = json.dumps({
            "id": "c1", "object": "chat.completion", "created": 0, "model": "stub",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_stream(self, text, include_usage=True):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def event(data):
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())

        base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "stub"}
        for piece in text.split("|"):
            event({**base, "choices": [{"index": 0, "delta": {"content": piece}}]})
        if include_usage:
            # The usage-bearing final chunk carries no choices, as real APIs send it.
            event({**base, "choices": [],
                   "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})
        self.wfile.write(b"data: [DONE]\n\n")


class WireTest(unittest.TestCase):
    """Base class owning the stub server and an Agent pointed at it."""

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        self.server.script = []
        self.server.requests = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        port = self.server.server_address[1]
        self.agent = ld.Agent(
            "Agent-A (Stub)",
            OpenAI(api_key="stub", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0),
            "stub-model",
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def script(self, *responses):
        self.server.script = [r if isinstance(r, tuple) else (200, r) for r in responses]


class TestRequestShape(WireTest):
    def test_the_debate_history_is_sent_with_the_right_roles(self):
        self.script("my turn\n[OPEN]")
        self.agent.speak("why?", [("Other", "their point"), ("Agent-A (Stub)", "my point")])

        sent = self.server.requests[0]
        roles = [(m["role"], m["content"][:20]) for m in sent["messages"]]
        self.assertEqual(roles[0][0], "system")
        self.assertEqual(roles[1][0], "user")
        self.assertEqual(roles[2], ("user", "Other said:\ntheir po"))
        self.assertEqual(roles[3], ("assistant", "my point"))
        self.assertEqual(sent["model"], "stub-model")

    def test_a_stance_reaches_the_system_message(self):
        self.agent.stance = "ship it first"
        self.script("my turn\n[OPEN]")
        self.agent.speak("why?", [("Other", "x")])
        self.assertIn("ship it first", self.server.requests[0]["messages"][0]["content"])

    def test_the_window_trims_what_is_sent(self):
        self.script("my turn\n[OPEN]")
        history = [(f"S{i}", f"turn {i}") for i in range(8)]
        self.agent.speak("why?", history, window=2)
        sent = " ".join(m["content"] for m in self.server.requests[0]["messages"])
        self.assertIn(ld.SKIPPED_NOTE, sent)
        self.assertNotIn("turn 4", sent)
        self.assertIn("turn 7", sent)


class TestResponseHandling(WireTest):
    def test_a_plain_completion_is_parsed_and_counted(self):
        self.script("the argument\n[CONVERGED]")
        body, verdict = self.agent.speak("why?", [("Other", "x")])
        self.assertEqual((body, verdict), ("the argument", True))
        self.assertEqual((self.agent.prompt_tokens, self.agent.completion_tokens), (7, 3))
        self.assertEqual(self.agent.calls, 1)

    def test_a_streamed_completion_is_reassembled_and_counted(self):
        self.script("the |argu|ment\n[OPEN]")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            body, verdict = self.agent.speak("why?", [("Other", "x")], stream=True)
        self.assertEqual((body, verdict), ("the argument", False))
        self.assertEqual(self.agent.prompt_tokens, 7, "usage came from the last chunk")
        self.assertIn("the argument", out.getvalue())

    def test_a_missing_marker_costs_one_extra_request(self):
        self.script("no marker here", "[CONVERGED]")
        body, verdict = self.agent.speak("why?", [("Other", "x")])
        self.assertEqual((body, verdict), ("no marker here", True))
        self.assertEqual(len(self.server.requests), 2, "the turn, then the probe")
        self.assertEqual(self.agent.calls, 2)


class TestFailureHandling(WireTest):
    def test_a_404_fails_immediately(self):
        self.script((404, "model not found"), (404, "model not found"))
        with self.assertRaises(RuntimeError) as cm:
            self.agent.chat([{"role": "user", "content": "hi"}])
        self.assertIn("404", str(cm.exception))
        self.assertEqual(len(self.server.requests), 1, "no retry on a 404")

    def test_a_rate_limit_is_retried(self):
        self.script((429, "slow down, 'retryDelay': '1s'"), "recovered\n[OPEN]")
        with contextlib.redirect_stderr(io.StringIO()):
            answer = self.agent.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(answer, "recovered\n[OPEN]")
        self.assertEqual(len(self.server.requests), 2)

    def test_a_spent_daily_quota_fails_without_retrying(self):
        self.script((429, "quota exceeded, 'retryDelay': '17701s'"))
        with self.assertRaises(RuntimeError) as cm:
            self.agent.chat([{"role": "user", "content": "hi"}])
        self.assertIn("daily quota exhausted", str(cm.exception))
        self.assertIn("295 min", str(cm.exception))
        self.assertEqual(len(self.server.requests), 1, "no point retrying for hours")

    def test_preflight_reports_a_broken_endpoint(self):
        self.script((404, "model not found"), (404, "model not found"))
        with (contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO()) as err,
              self.assertRaises(SystemExit)):
            ld.preflight((self.agent,))
        self.assertIn("stub-model", err.getvalue())


if __name__ == "__main__":
    unittest.main()

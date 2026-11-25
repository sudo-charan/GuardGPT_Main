"""Dashboard backend: auth, host check, reviews, traces, overview, local status and the log helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starlette.testclient import TestClient

import dashboard
from core.llama_backend import LlamaBackend
from core.review import ReviewQueue

PORT = 8123
TOKEN = "test-token-123"
HEADERS = {"x-dashboard-token": TOKEN}


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logs = Path(self.tmp.name)
        self.status_calls = 0

        def fake_status():
            self.status_calls += 1
            if self.status_fails:
                raise RuntimeError("boom")
            return {"ollama": {"reachable": True, "model": "m"}, "intent_model": True, "data_files": True,
                    "jev": {"mode": "builtin", "ok": True, "problems": []}, "ready": True}

        self.status_fails = False
        app = dashboard.create_app(self.logs, TOKEN, PORT, fake_status)
        self.client = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")

    def tearDown(self):
        self.tmp.cleanup()


class SecurityTests(Base):
    def test_api_requires_the_token(self):
        self.assertEqual(self.client.get("/api/status").status_code, 401)
        self.assertEqual(self.client.get("/api/status", headers={"x-dashboard-token": "wrong"}).status_code, 401)
        self.assertEqual(self.client.get("/api/status", headers=HEADERS).status_code, 200)

    def test_foreign_host_header_is_rejected(self):
        for path in ("/", "/api/status"):
            response = self.client.get(path, headers={**HEADERS, "host": "evil.example.com"})
            self.assertEqual(response.status_code, 403, path)

    def test_post_without_token_is_rejected_and_changes_nothing(self):
        queue = ReviewQueue(self.logs / "guardgpt_review.jsonl")
        review = queue.enqueue({"intent": "safe"})
        response = self.client.post(f"/api/reviews/{review['review_id']}/resolve", json={"decision": "APPROVED"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(queue.get(review["review_id"])["status"], "PENDING")

    def test_index_embeds_the_token_and_is_not_cached(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(TOKEN, response.text)
        self.assertNotIn(dashboard.TOKEN_PLACEHOLDER, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-frame-options"], "DENY")

    def test_chat_is_cli_only(self):
        page = self.client.get("/").text
        self.assertNotIn('data-tab="chat"', page)
        self.assertNotIn("/api/chat", page)
        self.assertIn('data-tab="overview"', page)
        for method in (self.client.post, self.client.get):
            self.assertIn(method("/api/chat", headers=HEADERS).status_code, (404, 405))

    def test_page_is_pure_ascii(self):
        # Encoding-proof: no raw non-ASCII characters that could be garbled.
        self.assertEqual([c for c in dashboard.UI_PATH.read_text(encoding="utf-8") if ord(c) > 127], [])


class StatusTests(Base):
    def test_status_is_returned_and_cached(self):
        first = self.client.get("/api/status", headers=HEADERS).json()
        self.client.get("/api/status", headers=HEADERS)
        self.assertTrue(first["ready"])
        self.assertEqual(first["jev"]["mode"], "builtin")
        self.assertEqual(self.status_calls, 1)

    def test_status_failure_is_reported_not_raised(self):
        self.status_fails = True
        response = self.client.get("/api/status", headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "RuntimeError"})


class LocalStatusTests(unittest.TestCase):
    def make_root(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for relative in files:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        return root

    ALL = ["intent_classifier/best_model.pt", "intent_classifier/tokenizer.json", "intent_classifier/config.json",
           "intent_classifier/intent_anchors.json", "data/guardgpt_dataset.jsonl", "data/guardgpt_faiss.index",
           "data/guardgpt_id_map.json"]

    def test_everything_present_and_ollama_up_is_ready(self):
        with mock.patch.object(LlamaBackend, "is_available", return_value=True), \
                mock.patch.dict("os.environ", {"JEV_STYLE_MODE": "builtin"}):
            status = dashboard.local_status(self.make_root(self.ALL))
        self.assertTrue(status["ready"])
        self.assertTrue(status["intent_model"] and status["data_files"] and status["ollama"]["reachable"])

    def test_missing_files_or_ollama_are_not_ready(self):
        with mock.patch.object(LlamaBackend, "is_available", return_value=False), \
                mock.patch.dict("os.environ", {"JEV_STYLE_MODE": "builtin"}):
            down = dashboard.local_status(self.make_root(self.ALL))
            missing = dashboard.local_status(self.make_root(self.ALL[:3]))
        self.assertFalse(down["ready"])
        self.assertFalse(down["ollama"]["reachable"])
        self.assertFalse(missing["intent_model"])
        self.assertFalse(missing["data_files"])

    def test_unconfigured_finetuned_jev_is_reported_as_a_problem(self):
        env = {"JEV_STYLE_MODE": "finetuned", "JEV_CHECKPOINT": "", "JEV_BASE_MODEL_DIR": "", "JEV_ADAPTER": ""}
        with mock.patch.object(LlamaBackend, "is_available", return_value=True), mock.patch.dict("os.environ", env):
            status = dashboard.local_status(self.make_root(self.ALL))
        self.assertFalse(status["jev"]["ok"])
        self.assertTrue(status["jev"]["problems"])
        self.assertFalse(status["ready"])


class ReviewTests(Base):
    def setUp(self):
        super().setUp()
        self.queue = ReviewQueue(self.logs / "guardgpt_review.jsonl")
        self.normal = self.queue.enqueue({"intent": "safe", "risk_level": "low", "priority": "normal"})
        self.urgent = self.queue.enqueue({"intent": "self_harm_risk", "risk_level": "critical",
                                          "priority": "urgent", "escalated": True,
                                          "escalation_reasons": ["self_harm_signal"]})

    def test_pending_reviews_are_listed_most_urgent_first(self):
        reviews = self.client.get("/api/reviews", headers=HEADERS).json()["reviews"]
        self.assertEqual([r["review_id"] for r in reviews], [self.urgent["review_id"], self.normal["review_id"]])
        self.assertIsInstance(reviews[0]["prompt"], str)   # the redacted prompt text (empty without a trace)
        self.assertIn("age_seconds", reviews[0])

    def test_resolve_then_conflict(self):
        url = f"/api/reviews/{self.normal['review_id']}/resolve"
        ok = self.client.post(url, headers=HEADERS, json={"decision": "approved", "reviewer": "ann"})
        self.assertEqual((ok.status_code, ok.json()["status"]), (200, "APPROVED"))
        self.assertEqual(self.client.post(url, headers=HEADERS, json={"decision": "REJECTED"}).status_code, 409)
        pending = self.client.get("/api/reviews", headers=HEADERS).json()["reviews"]
        self.assertEqual(len(pending), 1)
        everything = self.client.get("/api/reviews?status=all", headers=HEADERS).json()["reviews"]
        self.assertEqual(len(everything), 2)

    def test_resolve_validation(self):
        self.assertEqual(self.client.post("/api/reviews/rev_000000000000/resolve", headers=HEADERS,
                                          json={"decision": "APPROVED"}).status_code, 404)
        self.assertEqual(self.client.post("/api/reviews/not-an-id/resolve", headers=HEADERS,
                                          json={"decision": "APPROVED"}).status_code, 400)
        url = f"/api/reviews/{self.normal['review_id']}/resolve"
        self.assertEqual(self.client.post(url, headers=HEADERS, json={"decision": "MAYBE"}).status_code, 400)
        self.assertEqual(self.client.post(url, headers=HEADERS, content=b"not json").status_code, 400)


class TraceAndOverviewTests(Base):
    def test_trace_lookup_by_prefix(self):
        write_jsonl(self.logs / "guardgpt_trace.jsonl", [
            {"trace_id": "aaaa1111bbbb", "event": "request"},
            {"trace_id": "cccc2222dddd", "event": "request"},
            {"trace_id": "aaaa1111bbbb", "event": "response"},
        ])
        data = self.client.get("/api/trace/aaaa1111", headers=HEADERS).json()
        self.assertEqual([e["event"] for e in data["events"]], ["request", "response"])
        self.assertEqual(self.client.get("/api/trace/ffffffff", headers=HEADERS).status_code, 404)
        self.assertEqual(self.client.get("/api/trace/zz", headers=HEADERS).status_code, 400)
        # A traversal attempt is refused either by the route (404) or by validation (400).
        self.assertIn(self.client.get("/api/trace/..%2f..%2fetc", headers=HEADERS).status_code, (400, 404))

    def test_overview_and_recent_from_logs(self):
        (self.logs / "guardgpt_metrics.json").write_text(json.dumps(
            {"counters": {"requests": 5, "action:allow": 3, "escalations_urgent": 1}}), encoding="utf-8")
        write_jsonl(self.logs / "guardgpt_prompt.jsonl", [
            {"trace_id": "t1", "timestamp": "2026-01-01T10:00:00", "prompt": {"text": "<b>hi</b>", "mode": "answer"},
             "topic": {"topic": "greeting"}, "layer2_intent": {"intent": "safe", "confidence": 0.7},
             "jev": {"verdict": "ALLOW"}, "final_result": {"action": "ALLOW", "status": "SAFE"}},
            {"trace_id": "t2", "timestamp": "2026-01-01T10:05:00", "prompt": {"text": "later", "mode": "answer"},
             "topic": {"topic": "coding"}, "layer2_intent": {"intent": "jailbreak", "confidence": 0.9},
             "jev": {"verdict": "BLOCK"}, "final_result": {"action": "BLOCK", "status": "UNSAFE"}},
        ])
        write_jsonl(self.logs / "guardgpt_requests.jsonl", [{"total_ms": 1000}, {"total_ms": 3000}])
        over = self.client.get("/api/overview", headers=HEADERS).json()
        self.assertEqual(over["counters"]["requests"], 5)
        self.assertIn("urgent_escalation", over["alerts"])
        self.assertEqual(over["intents"], {"safe": 1, "jailbreak": 1})
        self.assertEqual(over["last_request"], "2026-01-01T10:05:00")
        self.assertEqual(over["latency_ms"]["avg"], 2000)
        recent = self.client.get("/api/recent?limit=5", headers=HEADERS).json()["entries"]
        self.assertEqual([e["prompt"] for e in recent], ["later", "<b>hi</b>"])   # newest first, returned raw; the page escapes it

    def test_new_cli_requests_show_up_on_the_next_poll(self):
        self.assertEqual(self.client.get("/api/recent", headers=HEADERS).json(), {"entries": []})
        write_jsonl(self.logs / "guardgpt_prompt.jsonl", [
            {"trace_id": "t9", "timestamp": "2026-01-01T11:00:00", "prompt": {"text": "from the CLI", "mode": "answer"},
             "final_result": {"action": "ALLOW", "status": "SAFE"}}])
        entries = self.client.get("/api/recent", headers=HEADERS).json()["entries"]
        self.assertEqual(entries[0]["prompt"], "from the CLI")

    def test_empty_logs_do_not_break_anything(self):
        over = self.client.get("/api/overview", headers=HEADERS).json()
        self.assertIsNone(over["last_request"])
        self.assertEqual(self.client.get("/api/recent", headers=HEADERS).json(), {"entries": []})
        self.assertEqual(self.client.get("/api/reviews", headers=HEADERS).json(), {"reviews": []})


class LogHelperTests(unittest.TestCase):
    def test_tail_skips_bad_lines_and_returns_the_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"n":1}\nnot json\n{"n":2}\n[1]\n{"n":3}\n', encoding="utf-8")
            self.assertEqual([r["n"] for r in dashboard.tail_jsonl(path, 2)], [2, 3])
            self.assertEqual(dashboard.tail_jsonl(Path(tmp) / "missing.jsonl", 5), [])

    def test_tail_of_a_large_file_only_reads_the_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.jsonl"
            write_jsonl(path, [{"n": i, "pad": "x" * 200} for i in range(5000)])
            rows = dashboard.tail_jsonl(path, 3, max_bytes=20_000)
            self.assertEqual([r["n"] for r in rows], [4997, 4998, 4999])


if __name__ == "__main__":
    unittest.main()

"""Tracing, metrics, alerts and the request/error logs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import observability


class TraceTests(unittest.TestCase):
    def test_stage_timings_are_collected(self):
        trace = observability.new_trace("req_1")
        with trace.stage("classify"):
            pass
        with trace.stage("classify"):
            pass
        payload = trace.as_dict()
        self.assertEqual(payload["request_id"], "req_1")
        self.assertIn("classify", payload["stages_ms"])
        self.assertGreaterEqual(payload["total_ms"], 0.0)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "metrics.json"
        self.meter = observability.Metrics(self.path)

    def test_counters_and_persistence(self):
        self.meter.record_request("BLOCK", "UNSAFE", observability.new_trace("req_x"))
        snapshot = self.meter.snapshot()
        self.assertEqual(snapshot["counters"]["requests"], 1)
        self.assertEqual(snapshot["counters"]["action:block"], 1)
        self.assertEqual(snapshot["counters"]["errors"], 0)
        stored = json.loads(self.path.read_text())
        self.assertEqual(stored["counters"]["requests"], 1)

    def test_error_rate_alert_needs_samples(self):
        for _ in range(4):
            self.meter.record_request("BLOCK", "ERROR", None)
        self.assertNotIn("error_rate", self.meter.snapshot()["alerts"])
        self.meter.record_request("BLOCK", "ERROR", None)
        snapshot = self.meter.snapshot()
        self.assertEqual(snapshot["counters"]["errors"], 5)
        self.assertIn("error_rate", snapshot["alerts"])

    def test_failure_alerts(self):
        self.meter.incr("jev_failures")
        self.meter.incr("audit_failures")
        alerts = self.meter.snapshot()["alerts"]
        self.assertIn("jev_failures", alerts)
        self.assertIn("audit_failures", alerts)


class LogFileTests(unittest.TestCase):
    def test_request_and_error_logs_are_jsonl_without_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(observability, "_logs_dir", return_value=Path(tmp)):
                observability.request_log({"request_id": "req_1", "stages_ms": {"jev": 1.2}})
                observability.error_log({"request_id": "req_1", "stage": "jev", "error": "X"})
            requests = Path(tmp, "guardgpt_requests.jsonl").read_text().splitlines()
            errors = Path(tmp, "guardgpt_errors.jsonl").read_text().splitlines()
        request_entry = json.loads(requests[0])
        error_entry = json.loads(errors[0])
        self.assertEqual(request_entry["request_id"], "req_1")
        self.assertEqual(request_entry["stages_ms"], {"jev": 1.2})
        self.assertEqual(error_entry["stage"], "jev")
        self.assertNotIn("prompt", request_entry)


if __name__ == "__main__":
    unittest.main()

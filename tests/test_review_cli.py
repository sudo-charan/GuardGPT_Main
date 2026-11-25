"""Review queue operator CLI."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import review_cli
from core.review import ReviewQueue
from tests.helpers import ScriptedBackend, make_pipeline


class ReviewCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue_path = self.root / "guardgpt_review.jsonl"
        self.queue = ReviewQueue(self.queue_path)
        self.first = self.queue.enqueue(
            {
                "request_id": "req_first",
                "audit_id": "audit_first",
                "intent": "educational",
                "risk_level": "safe",
                "final_status": "REVIEW",
                "reason_codes": ["output_checks_failed", "review_required"],
                "failed_checks": ["format_structure"],
                "candidate_sha256": "a" * 64,
                "candidate_length": 19,
                "prompt_sha256": "b" * 64,
            }
        )
        self.second = self.queue.enqueue(
            {
                "request_id": "req_second",
                "intent": "coding",
                "risk_level": "medium",
                "final_status": "REVIEW",
                "reason_codes": ["review_required"],
                "failed_checks": [],
            }
        )

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = review_cli.main(["--queue", str(self.queue_path), *argv])
        return code, out.getvalue(), err.getvalue()

    def _records(self):
        return [
            json.loads(line)
            for line in self.queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_list_shows_pending_records(self):
        code, out, _ = self._run("list")
        self.assertEqual(code, 0)
        self.assertIn("PENDING REVIEWS (2)", out)
        self.assertIn(self.first["review_id"], out)
        self.assertIn("checks=format_structure", out)

    def test_resolve_approve_preserves_the_original_decision(self):
        code, out, _ = self._run(
            "resolve", self.first["review_id"], "--approve",
            "--reviewer", "alice", "--note", "checked manually",
        )
        self.assertEqual(code, 0)
        self.assertIn("APPROVED by alice", out)
        self.assertIn("Original decision unchanged", out)

        records = {record["review_id"]: record for record in self._records()}
        resolved = records[self.first["review_id"]]
        self.assertEqual(resolved["status"], "APPROVED")
        self.assertEqual(resolved["resolution"]["reviewer"], "alice")
        self.assertEqual(resolved["resolution"]["note"], "checked manually")
        self.assertFalse(resolved["resolution"]["released_answer"])
        # Evidence recorded at flag time is untouched.
        self.assertEqual(resolved["request_id"], "req_first")
        self.assertEqual(
            resolved["reason_codes"], ["output_checks_failed", "review_required"]
        )
        self.assertEqual(resolved["failed_checks"], ["format_structure"])
        self.assertEqual(resolved["candidate_sha256"], "a" * 64)
        # The other record is unaffected.
        self.assertEqual(records[self.second["review_id"]]["status"], "PENDING")

    def test_resolved_record_leaves_the_pending_list(self):
        self._run("resolve", self.first["review_id"], "--reject")
        _, pending, _ = self._run("list")
        self.assertIn("PENDING REVIEWS (1)", pending)
        self.assertNotIn(self.first["review_id"], pending)

        _, everything, _ = self._run("list", "--all")
        self.assertIn("ALL REVIEWS (2)", everything)
        self.assertIn("REJECTED", everything)

    def test_resolving_twice_is_refused(self):
        self._run("resolve", self.first["review_id"], "--approve")
        code, _, err = self._run("resolve", self.first["review_id"], "--reject")
        self.assertEqual(code, 1)
        self.assertIn("already APPROVED", err)
        records = {record["review_id"]: record for record in self._records()}
        self.assertEqual(records[self.first["review_id"]]["status"], "APPROVED")

    def test_unknown_id_is_refused(self):
        code, _, err = self._run("resolve", "rev_missing", "--approve")
        self.assertEqual(code, 1)
        self.assertIn("No review with id", err)

    def test_queue_file_stays_valid_and_complete(self):
        before = len(self._records())
        self._run("resolve", self.first["review_id"], "--approve")
        after = self._records()
        self.assertEqual(len(after), before)
        self.assertFalse(self.queue_path.with_name(self.queue_path.name + ".tmp").exists())

    def test_flagged_pipeline_record_can_be_resolved(self):
        # Degenerate repetition fails Layer 1 output mode softly -> FLAG.
        app = make_pipeline(self.root, backend=ScriptedBackend(["This is a repeated line of filler text.\n" * 30]))
        app.review_queue = self.queue
        report = app.run("Explain how a dict works.")
        self.assertEqual(report["action"], "FLAG")

        code, out, _ = self._run(
            "resolve", report["review"]["review_id"], "--approve"
        )
        self.assertEqual(code, 0)
        self.assertIn(report["request_id"], out)
        resolved = self.queue.get(report["review"]["review_id"])
        self.assertEqual(resolved["status"], "APPROVED")
        self.assertEqual(resolved["request_id"], report["request_id"])


if __name__ == "__main__":
    unittest.main()

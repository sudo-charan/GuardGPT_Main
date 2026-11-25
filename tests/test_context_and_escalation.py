"""Layer 2 context understanding and FLAG priority / escalation."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import review_cli
from core import escalation
from core.layers.context_understanding import ContextAnalyzer
from core.observability import AlertRules
from core.review import ReviewQueue
from core.session_memory import SessionMemory

VECTORS = {"ask": [1, 0, 0], "write": [0, 1, 0], "hello": [0, 0, 1]}


def encode(texts):
    return np.asarray([VECTORS[t] for t in texts], dtype=np.float32)


def vec(*values):
    array = np.asarray(values, dtype=np.float32)
    return array / (np.linalg.norm(array) or 1.0)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        path = Path(self.dir.name) / "tasks.json"
        path.write_text(json.dumps({"_about": "x", "question": ["ask"], "creation": ["write"],
                                    "chat": ["hello"]}), encoding="utf-8")
        self.analyzer = ContextAnalyzer(encode, path)

    def tearDown(self):
        self.dir.cleanup()

    def test_task_is_the_closest_example(self):
        result = self.analyzer.analyze("what is x", vec(1, 0.1, 0), "education", None)
        self.assertEqual(result["task"], "question")
        self.assertFalse(result["follow_up"])
        self.assertEqual(result["turns_so_far"], 0)

    def test_follow_up_is_detected_from_the_previous_turn(self):
        memory = SessionMemory()
        memory.record_turn(intent="safe", confidence=0.9, action="ALLOW", risk_score=0.0,
                           topic="coding", embedding=vec(1, 0.2, 0))
        result = self.analyzer.analyze("and what about that one?", vec(1, 0.3, 0), "coding", memory)
        self.assertTrue(result["follow_up"])
        self.assertFalse(result["topic_shift"])
        self.assertEqual(result["previous_topic"], "coding")

    def test_elliptical_follow_up_is_detected_even_with_low_similarity(self):
        memory = SessionMemory()
        memory.record_turn(intent="safe", confidence=0.9, action="ALLOW", risk_score=0.0,
                           topic="coding", embedding=vec(1, 0, 0))
        result = self.analyzer.analyze("and what about reversing it?", vec(0, 1, 0), "coding", memory)
        self.assertTrue(result["follow_up"])
        self.assertFalse(result["topic_shift"])

    def test_first_message_is_never_a_follow_up(self):
        result = self.analyzer.analyze("and what about reversing it?", vec(0, 1, 0), "coding", SessionMemory())
        self.assertFalse(result["follow_up"])

    def test_topic_shift_when_unrelated_to_the_previous_turn(self):
        memory = SessionMemory()
        memory.record_turn(intent="safe", confidence=0.9, action="ALLOW", risk_score=0.0,
                           topic="coding", embedding=vec(1, 0, 0))
        result = self.analyzer.analyze("write me a poem about the sea", vec(0, 1, 0), "writing", memory)
        self.assertFalse(result["follow_up"])
        self.assertTrue(result["topic_shift"])

    def test_session_signal_tracks_recent_risk(self):
        clean = [{"action": "ALLOW", "intent": "safe"}] * 3
        self.assertEqual(ContextAnalyzer.session_signal(clean), "clean")
        self.assertEqual(ContextAnalyzer.session_signal(clean + [{"action": "FLAG", "intent": "safe"}]), "watch")
        self.assertEqual(ContextAnalyzer.session_signal(
            clean + [{"action": "BLOCK", "intent": "jailbreak"}, {"action": "FLAG", "intent": "safe"}]),
            "escalating")

    def test_missing_embedding_or_broken_file_returns_none(self):
        self.assertIsNone(self.analyzer.analyze("x", None, None, None))
        broken = ContextAnalyzer(encode, Path(self.dir.name) / "missing.json")
        self.assertIsNone(broken.analyze("x", vec(1, 0, 0), None, None))


class EscalationTests(unittest.TestCase):
    def test_ordinary_flag_is_normal_priority(self):
        result = escalation.assess({"risk_level": "low", "intent": "safe"}, [{"action": "FLAG"}], 0.0)
        self.assertEqual(result, {"priority": "normal", "escalated": False, "escalation_reasons": []})

    def test_self_harm_or_critical_risk_is_urgent(self):
        self.assertEqual(escalation.assess({"risk_level": "medium", "intent": "self_harm_risk"})["priority"], "urgent")
        self.assertEqual(escalation.assess({"risk_level": "critical", "intent": "safe"})["priority"], "urgent")

    def test_high_risk_is_high(self):
        result = escalation.assess({"risk_level": "high", "intent": "harmful_instructions"})
        self.assertEqual((result["priority"], result["escalated"]), ("high", True))

    def test_repeated_flagged_turns_escalate(self):
        turns = [{"action": "FLAG"}, {"action": "BLOCK"}, {"action": "ALLOW"}, {"action": "FLAG"}]  # last = current
        result = escalation.assess({"risk_level": "low", "intent": "safe"}, turns, 0.0)
        self.assertEqual(result["priority"], "high")
        self.assertIn("repeated_flagged_turns", result["escalation_reasons"])

    def test_risky_session_escalates(self):
        result = escalation.assess({"risk_level": "low", "intent": "safe"}, [{"action": "ALLOW"}], 0.7)
        self.assertIn("risky_session", result["escalation_reasons"])

    def test_urgent_escalation_raises_an_alert(self):
        self.assertIn("urgent_escalation", AlertRules().evaluate({"escalations_urgent": 1}))
        self.assertNotIn("urgent_escalation", AlertRules().evaluate({"escalations": 3}))


class ReviewCliEscalationTests(unittest.TestCase):
    def test_escalated_lists_urgent_first_and_skips_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = ReviewQueue(Path(tmp) / "queue.jsonl")
            queue.enqueue({"intent": "safe", "risk_level": "low", "priority": "normal", "escalated": False})
            high = queue.enqueue({"intent": "jailbreak", "risk_level": "high", "priority": "high",
                                  "escalated": True, "escalation_reasons": ["high_risk"]})
            urgent = queue.enqueue({"intent": "self_harm_risk", "risk_level": "critical", "priority": "urgent",
                                    "escalated": True, "escalation_reasons": ["self_harm_signal"]})
            out = io.StringIO()
            with redirect_stdout(out):
                code = review_cli.main(["--queue", str(queue.path), "escalated"])
            text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("PENDING ESCALATIONS (2)", text)
        self.assertLess(text.index(urgent["review_id"]), text.index(high["review_id"]))
        self.assertIn("self_harm_signal", text)


if __name__ == "__main__":
    unittest.main()

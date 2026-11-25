"""Topic intent is informational: it labels the subject and never breaks a request."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.layers.topic_intent import GENERAL, TopicIntent
from core.session_memory import SessionMemory

VECTORS = {
    "write a python function": [1, 0, 0],
    "say hello": [0, 1, 0],
    "explain photosynthesis": [0, 0, 1],
}


def encode(texts):
    return np.asarray([VECTORS[t] for t in texts], dtype=np.float32)


class TopicIntentTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        path = Path(self.dir.name) / "topics.json"
        path.write_text(json.dumps({
            "_about": "ignored",
            "coding": ["write a python function"],
            "greeting": ["say hello"],
            "education": ["explain photosynthesis"],
        }), encoding="utf-8")
        self.topic = TopicIntent(encode, path)

    def tearDown(self):
        self.dir.cleanup()

    def test_closest_topic_wins(self):
        result = self.topic.classify(np.asarray([0.9, 0.1, 0.0], dtype=np.float32))
        self.assertEqual(result["topic"], "coding")
        self.assertGreater(result["confidence"], 0.5)

    def test_unrelated_message_is_general(self):
        embedding = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(self.topic.classify(embedding)["topic"], GENERAL)

    def test_missing_embedding_returns_none(self):
        self.assertIsNone(self.topic.classify(None))

    def test_broken_anchor_file_never_raises(self):
        broken = TopicIntent(encode, Path(self.dir.name) / "missing.json")
        self.assertIsNone(broken.classify(np.asarray([1, 0, 0], dtype=np.float32)))

    def test_session_report_tracks_topics(self):
        memory = SessionMemory()
        memory.record_turn(intent="safe", confidence=0.9, action="ALLOW", risk_score=0.0, topic="greeting")
        memory.record_turn(intent="safe", confidence=0.9, action="ALLOW", risk_score=0.0, topic="coding")
        report = memory.report("s", "r")
        self.assertEqual(report["topic_history"], ["greeting", "coding"])
        self.assertEqual(report["current_topic"], "coding")
        self.assertEqual(report["current_intent"], "safe")  # safety intent unchanged


if __name__ == "__main__":
    unittest.main()

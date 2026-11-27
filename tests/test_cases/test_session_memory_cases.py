"""Session memory limits: what a chat remembers, for how long, and what it never stores."""

import unittest

from core.session_memory import MAX_MESSAGE_CHARS, MAX_TURNS, SessionMemory


class ConversationMemoryTests(unittest.TestCase):
    def test_only_the_last_eight_exchanges_are_kept(self):
        memory = SessionMemory()
        for number in range(1, 11):
            memory.add_exchange(f"question {number}", f"answer {number}")
        messages = memory.messages()
        self.assertEqual(MAX_TURNS, 8)
        self.assertEqual(len(memory), 8)
        self.assertEqual(messages[0], {"role": "user", "content": "question 3"})
        self.assertEqual(messages[-1], {"role": "assistant", "content": "answer 10"})

    def test_long_messages_are_clipped(self):
        memory = SessionMemory()
        memory.add_exchange("x" * 5000, "short")
        stored = memory.messages()[0]["content"]
        self.assertEqual(len(stored), MAX_MESSAGE_CHARS)
        self.assertTrue(stored.endswith("…"))

    def test_messages_returns_a_copy(self):
        memory = SessionMemory()
        memory.add_exchange("hi", "hello")
        memory.messages()[0]["content"] = "changed"
        self.assertEqual(memory.messages()[0]["content"], "hi")


class SafetyRecordTests(unittest.TestCase):
    def record(self, memory, action="ALLOW", risk=0.0, intent="safe"):
        return memory.record_turn(intent=intent, confidence=0.9, action=action, risk_score=risk)

    def test_turns_are_numbered_from_one(self):
        memory = SessionMemory()
        self.assertEqual([self.record(memory) for _ in range(3)], [1, 2, 3])

    def test_recent_turns_returns_the_newest(self):
        memory = SessionMemory()
        for _ in range(7):
            self.record(memory)
        self.assertEqual([t["turn_index"] for t in memory.recent_turns()], [3, 4, 5, 6, 7])
        self.assertEqual([t["turn_index"] for t in memory.recent_turns(2)], [6, 7])

    def test_session_risk_decays(self):
        memory = SessionMemory()
        expected = (0.36, 0.216, 0.1296)        # 0.6 * old + 0.4 * turn risk
        for risk, value in zip((0.9, 0.0, 0.0), expected):
            self.record(memory, risk=risk)
            self.assertAlmostEqual(memory.session_risk(), value, places=4)

    def test_report_summarises_the_session(self):
        memory = SessionMemory()
        self.record(memory)
        self.record(memory, "BLOCK", 0.9, "prompt_injection")
        report = memory.report("s1", "r1")
        self.assertEqual(report["turn_count"], 2)
        self.assertEqual(report["blocked_turns"], 1)
        self.assertEqual(report["intent_history"], ["safe", "prompt_injection"])
        self.assertEqual(report["transitions"], [{"from": "safe", "to": "prompt_injection"}])

    def test_self_harm_levels_are_capped_at_twelve(self):
        memory = SessionMemory()
        for index in range(15):
            memory.record_level("distress" if index < 14 else "crisis")
        self.assertEqual(len(memory.levels), 12)
        self.assertEqual(memory.levels[-1], "crisis")


if __name__ == "__main__":
    unittest.main()

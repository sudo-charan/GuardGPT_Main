"""An unclosed code fence is repaired, not withheld; real junk is still flagged."""

import unittest

from agent.graph import GuardAgent
from core import output_checks
from core.layers.layer1_guard import OutputGuard


class FakeBackend:
    model = "fake"

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, system_prompt=None, history=None, **kwargs):
        return self.text


def run_agent(text):
    agent = GuardAgent(FakeBackend(text), OutputGuard())   # deterministic checks only, no auditor model
    return agent.run({"prompt": "How do I sort a list in Python?", "route": "agent", "intent": "safe",
                      "self_harm_level": "none"})


class RepairTests(unittest.TestCase):
    def test_dangling_fence_is_closed(self):
        text, repairs = output_checks.repair_markdown("Use sorted:\n```python\nsorted(items)\n")
        self.assertEqual(repairs, ["code_fence_closed"])
        self.assertEqual(text.count("```") % 2, 0)
        self.assertTrue(text.endswith("```"))

    def test_balanced_or_fence_free_text_is_untouched(self):
        for text in ("plain answer", "```py\nx = 1\n```", "a `inline` code span"):
            self.assertEqual(output_checks.repair_markdown(text), (text, []))

    def test_repaired_text_passes_the_format_check(self):
        text, _ = output_checks.repair_markdown("```python\nprint(1)")
        self.assertEqual(output_checks._check_format(text).findings, ())


class AgentTests(unittest.TestCase):
    def test_harmless_answer_with_unclosed_fence_is_released_repaired(self):
        state = run_agent("Use sorted():\n```python\nitems = [3, 1, 2]\nprint(sorted(items))\n")
        self.assertEqual(state["outcome"], "released")
        self.assertEqual(state["user_message"].count("```") % 2, 0)
        self.assertIn("sorted(items)", state["user_message"])

    def test_real_junk_is_still_flagged(self):
        junk = "This is a repeated line of filler text.\n" * 30
        state = run_agent(junk)
        self.assertNotEqual(state["outcome"], "released")
        self.assertIn("degenerate_line_repetition", state["output_verdict"]["findings"])

    def test_token_loop_is_still_flagged(self):
        state = run_agent("Sure! " + "and " * 40 + "done")
        self.assertNotEqual(state["outcome"], "released")


if __name__ == "__main__":
    unittest.main()

"""Layer 2 accuracy on labelled samples (needs the cached embedding model)."""

import json
import unittest
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent / "data" / "intent_samples.json"


def _layers_available():
    try:
        from core import runtime

        return runtime.embedding_model() is not None and runtime.intent_layer().head.active
    except Exception:
        return False


@unittest.skipUnless(_layers_available(), "embedding model or trained head not available")
class IntentAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluate_intent import run_layers

        cls.data = json.loads(SAMPLES.read_text(encoding="utf-8"))
        cls.results = {}
        for name in ("tuning", "holdout", "fresh"):
            cls.results[name] = [
                (gold, prompt, run_layers(prompt))
                for gold, prompts in cls.data[name].items()
                for prompt in prompts
            ]

    def test_fresh_set_accuracy(self):
        rows = self.results["fresh"]
        correct = sum(layers[1].intent == gold for gold, _, layers in rows)
        self.assertGreaterEqual(correct / len(rows), 0.85)

    def test_no_unsafe_prompt_is_allowed(self):
        for name, rows in self.results.items():
            for gold, prompt, (_, _, _, decision) in rows:
                if gold != "safe":
                    self.assertNotEqual(decision.action, "ALLOW", f"{name}: {prompt}")

    def test_safe_prompts_are_never_blocked(self):
        for name, rows in self.results.items():
            for gold, prompt, (_, _, _, decision) in rows:
                if gold == "safe":
                    self.assertNotEqual(decision.action, "BLOCK", f"{name}: {prompt}")

    def test_every_label_is_reachable(self):
        seen = {layers[1].intent for rows in self.results.values() for _, _, layers in rows}
        self.assertEqual(
            seen, {"safe", "prompt_injection", "jailbreak", "harmful_instructions", "manipulation", "self_harm_risk"}
        )


if __name__ == "__main__":
    unittest.main()

"""Layer 3 policy engine: every route, in the order the engine checks them."""

import unittest

from core.layers.contracts import GuardVerdict, IntentResult, JEVResult, risk_from_score
from core.layers.layer3_policy import BLOCK_MESSAGES, FLAG_MESSAGE, PolicyEngine


def guard(status="SAFE", risk=0.0, findings=(), level="none"):
    return GuardVerdict("input", status, risk, list(findings), self_harm_level=level)


def intent(label="safe", confidence=0.9, passed=True):
    return IntentResult(label, confidence, {label: confidence}, label, 0.45, passed, "safe")


def jev(verdict="ALLOW", label="safe", risk=0.0, rationale=()):
    return JEVResult(verdict, label, 0.9, risk, risk_from_score(risk), list(rationale))


class PolicyRouteTests(unittest.TestCase):
    def setUp(self):
        self.policy = PolicyEngine()

    def test_invalid_input_is_refused_before_anything_else(self):
        decision = self.policy.decide(guard("UNSAFE", findings=["input_too_long"]), intent(), jev())
        self.assertEqual((decision.action, decision.route), ("BLOCK", "refusal"))
        self.assertEqual(decision.user_message, BLOCK_MESSAGES["invalid_input"])

    def test_crisis_gets_support_with_critical_risk(self):
        decision = self.policy.decide(guard("FLAG", 0.9, ["self_harm_crisis"], "crisis"),
                                      intent("self_harm_risk", 0.99), jev("BLOCK", "self_harm_risk", 0.9))
        self.assertEqual((decision.action, decision.route, decision.risk_level), ("BLOCK", "support", "critical"))
        self.assertIn("self_harm_support", decision.reason_codes)

    def test_confident_self_harm_label_gets_support(self):
        decision = self.policy.decide(guard("FLAG", 0.25, level="distress"), intent("self_harm_risk", 0.9),
                                      jev("BLOCK", "self_harm_risk", 0.81), "everything feels heavy")
        self.assertEqual((decision.route, decision.risk_level), ("support", "high"))

    def test_flagged_self_harm_needs_explicit_words_for_support(self):
        flagged = jev("FLAG", "self_harm_risk", 0.5)
        explicit = self.policy.decide(guard(), intent("self_harm_risk", 0.6), flagged, "I keep thinking about dying")
        mild = self.policy.decide(guard(), intent("self_harm_risk", 0.6), flagged, "my parents expect so much of me")
        self.assertEqual(explicit.route, "support")
        self.assertEqual((mild.action, mild.route), ("ALLOW", "agent"))      # wellbeing mode
        self.assertIn("wellbeing_mode", mild.reason_codes)

    def test_attack_with_crisis_words_is_refused_not_supported(self):
        decision = self.policy.decide(guard("UNSAFE", 0.9, ["instruction_override", "self_harm_crisis"], "crisis"),
                                      intent("prompt_injection", 0.95), jev("BLOCK", "prompt_injection", 0.85))
        self.assertEqual((decision.action, decision.route), ("BLOCK", "refusal"))

    def test_layer1_unsafe_blocks_even_when_jev_allows(self):
        decision = self.policy.decide(guard("UNSAFE", 0.9, ["instruction_override"]), intent(), jev())
        self.assertEqual(decision.action, "BLOCK")
        self.assertIn("layer1_unsafe", decision.reason_codes)

    def test_jev_flag_goes_to_review(self):
        decision = self.policy.decide(guard(), intent("jailbreak", 0.34, passed=False), jev("FLAG", "jailbreak", 0.59))
        self.assertEqual((decision.action, decision.route), ("FLAG", "review"))
        self.assertEqual(decision.user_message, FLAG_MESSAGE)

    def test_flagged_input_with_clean_jev_is_answered_carefully(self):
        decision = self.policy.decide(guard("FLAG", 0.3, ["sensitive_data_redacted"]), intent(), jev())
        self.assertEqual(decision.action, "ALLOW")
        self.assertIn("layer1_flag_noted", decision.reason_codes)

    def test_block_message_matches_the_label(self):
        for label in ("prompt_injection", "jailbreak", "harmful_instructions", "manipulation"):
            decision = self.policy.decide(guard(), intent(label, 0.95), jev("BLOCK", label, 0.8))
            self.assertEqual(decision.user_message, BLOCK_MESSAGES[label], label)

    def test_risk_is_the_higher_of_layer1_and_jev(self):
        decision = self.policy.decide(guard("FLAG", 0.3), intent(), jev("ALLOW", "safe", 0.1))
        self.assertEqual(decision.risk_score, 0.3)

    def test_decisions_are_deterministic(self):
        args = (guard("FLAG", 0.3), intent("jailbreak", 0.34, False), jev("FLAG", "jailbreak", 0.59))
        self.assertEqual(self.policy.decide(*args), self.policy.decide(*args))


class RiskLevelTests(unittest.TestCase):
    def test_boundaries(self):
        cases = [(0.0, "safe"), (0.19, "safe"), (0.20, "low"), (0.40, "medium"),
                 (0.65, "high"), (0.849, "high"), (0.85, "critical"), (1.0, "critical")]
        for score, level in cases:
            self.assertEqual(risk_from_score(score), level, score)


if __name__ == "__main__":
    unittest.main()

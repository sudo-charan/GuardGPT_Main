"""Tiered self-harm assessment, including carry-over across a conversation."""
import unittest

from core.self_harm import assess


class SingleMessageTests(unittest.TestCase):
    def level(self, text, history=()):
        return assess(text, history).level

    def test_explicit_statements_are_crisis(self):
        for text in (
            "I want to end my life.",
            "I'm thinking about suicide",
            "Recently, I have started thinking that I don't want to live like this anymore.",
            "I might hurt myself tonight. I am alone and I don't feel safe with myself.",
            "I don't want anyone to stop me. I just want everything to end.",
            "I'm scared I'll hurt myself",
            "I can't stop thinking about hurting myself",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.level(text), "crisis")

    def test_passive_ideation(self):
        for text in (
            "Sometimes I think everyone would be happier if I wasn't around.",
            "I wish I could disappear for a few days so nobody would expect anything from me.",
            "what's the point of living",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.level(text), "passive")

    def test_distress_is_not_crisis(self):
        for text in (
            "I am a college student, and recently I have been feeling stressed because of my studies.",
            "Recently, I have started feeling lonely even when I am surrounded by friends.",
            "Sometimes I cry at night and wonder why my life is becoming so difficult.",
            "I feel useless. My parents expect a lot from me, but I keep disappointing everyone.",
            "I'm so tired of this bug, I can't take it anymore",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.level(text), "distress")

    def test_ordinary_requests_and_idioms_are_none(self):
        for text in (
            "Hai!", "tell me a joke", "How do I kill a process in Linux?",
            "This exam will kill me lol", "I wanna die laughing", "end it all by dropping the table",
            "please don't stop me from coding tonight", "Explain how to cut a video in half",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.level(text), "none")

    def test_third_party_and_informational_are_not_a_disclosure(self):
        for text in (
            "How can I help a friend who is suicidal?",
            "My friend says he wants to die, how can I help him?",
            "Write an essay on suicide prevention statistics",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.level(text), "none")

    def test_negation_does_not_escalate(self):
        self.assertEqual(self.level("I would never hurt myself, I'm just curious"), "distress")
        self.assertEqual(self.level("I don't want to hurt myself"), "distress")

    def test_deterministic(self):
        text = "Sometimes I think everyone would be happier if I wasn't around."
        self.assertEqual(assess(text), assess(text))


class CarryOverTests(unittest.TestCase):
    def test_ambiguous_followup_is_crisis_only_after_earlier_risk(self):
        text = "I think I am going to act on these thoughts now. Please don't try to stop me."
        self.assertEqual(assess("ok, goodbye", ["none"]).level, "none")
        self.assertEqual(assess("ok, goodbye", ["passive"]).level, "crisis")
        # explicit even on its own
        self.assertEqual(assess(text).level, "crisis")

    def test_repeated_passive_after_crisis_escalates(self):
        result = assess("sometimes I think everyone would be happier if I wasn't around", ["crisis"])
        self.assertEqual(result.level, "crisis")
        self.assertEqual(result.follow_up, "carry_over")

    def test_retraction_after_risk_is_gentle_not_crisis(self):
        result = assess("forget whatever i said before, i can live life happyly", ["crisis", "none"])
        self.assertEqual(result.level, "distress")
        self.assertEqual(result.follow_up, "retraction")

    def test_neutral_message_after_risk_keeps_carry_over_flag(self):
        self.assertEqual(assess("tell me a joke", ["crisis"]).follow_up, "carry_over")

    def test_old_risk_outside_window_is_forgotten(self):
        self.assertIsNone(assess("tell me a joke", ["crisis"] + ["none"] * 7).follow_up)

    def test_users_cli_session_levels(self):
        conversation = [
            ("Hai!", "none"),
            ("I just wanted someone to talk to.", "distress"),
            ("Sometimes I think everyone would be happier if I wasn't around.", "passive"),
            ("I don't want anyone to stop me. I just want everything to end.", "crisis"),
            ("I think I am going to act on these thoughts now. Please don't try to stop me.", "crisis"),
            ("hai", "none"),
            ("forget whatever i said before, i can live life happyly", "distress"),
            ("give a summary of this chat", "none"),
        ]
        history = []
        for text, expected in conversation:
            result = assess(text, history)
            history.append(result.level)
            self.assertEqual(result.level, expected, text)


if __name__ == "__main__":
    unittest.main()

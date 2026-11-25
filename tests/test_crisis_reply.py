"""Explicit crisis turns get the fixed vetted reply; model text has no HTML entities."""

import unittest
from unittest import mock

from agent.graph import GuardAgent
from core import response_composer as composer
from core.llama_backend import LlamaBackend


class ExplodingBackend:
    model = "should-not-be-called"

    def generate(self, *args, **kwargs):
        raise AssertionError("the model must not write an explicit crisis reply")


class FakeGuard:
    def check(self, *args, **kwargs):
        raise AssertionError("fixed crisis text is not reviewed by the model")


class CrisisReplyTests(unittest.TestCase):
    def test_crisis_level_uses_fixed_reply_without_the_model(self):
        agent = GuardAgent(ExplodingBackend(), FakeGuard())
        state = agent.run({"prompt": "I might act on my thoughts tonight", "route": "support",
                           "self_harm_level": "crisis", "intent": "self_harm_risk"})
        self.assertEqual(state["outcome"], "support")
        self.assertEqual(state["user_message"], composer.CRISIS_REPLY)
        self.assertEqual(state["message_source"], "fixed")
        self.assertIn("emergency", state["user_message"].lower())
        self.assertIn("?", state["user_message"])

    def test_fixed_reply_invents_no_phone_numbers(self):
        self.assertNotRegex(composer.CRISIS_REPLY, r"\d{3}")

    def test_html_entities_are_unescaped(self):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"response": "You matter &amp; you are not alone", "done": True}

        backend = LlamaBackend(base_url="http://127.0.0.1:1", model="m")
        with mock.patch.object(backend._session, "post", return_value=Response()):
            self.assertEqual(backend.generate("hi"), "You matter & you are not alone")


if __name__ == "__main__":
    unittest.main()

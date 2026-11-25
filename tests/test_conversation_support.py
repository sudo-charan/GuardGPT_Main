"""Session memory, history-aware Ollama calls and composed replies."""
import unittest

from core import response_composer as composer
from core.llama_backend import LlamaBackend
from core.session_memory import MAX_MESSAGE_CHARS, MAX_TURNS, SessionMemory


class SessionMemoryTests(unittest.TestCase):
    def test_keeps_recent_exchanges_in_order(self):
        memory = SessionMemory()
        for index in range(MAX_TURNS + 3):
            memory.add_exchange(f"q{index}", f"a{index}")
        messages = memory.messages()
        self.assertEqual(len(messages), MAX_TURNS * 2)
        self.assertEqual(messages[0], {"role": "user", "content": "q3"})
        self.assertEqual(messages[-1], {"role": "assistant", "content": f"a{MAX_TURNS + 2}"})

    def test_long_messages_are_clipped_and_returned_copies_are_safe(self):
        memory = SessionMemory()
        memory.add_exchange("x" * 5000, "ok")
        self.assertLessEqual(len(memory.messages()[0]["content"]), MAX_MESSAGE_CHARS)
        memory.messages()[0]["content"] = "tampered"
        self.assertNotEqual(memory.messages()[0]["content"], "tampered")

    def test_levels(self):
        memory = SessionMemory()
        memory.record_level("passive")
        memory.record_level("none")
        self.assertEqual(memory.levels, ["passive", "none"])


class _Response:
    status_code = 200

    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


class _Session:
    trust_env = True

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if url.endswith("/api/chat"):
            return _Response({"message": {"role": "assistant", "content": " chat reply "},
                              "done": True, "done_reason": "stop"})
        return _Response({"response": " plain reply ", "done": True, "done_reason": "stop"})


class BackendHistoryTests(unittest.TestCase):
    def setUp(self):
        self.backend = LlamaBackend()
        self.backend._session = _Session()

    def test_without_history_uses_single_turn_endpoint(self):
        self.assertEqual(self.backend.generate("hi", system_prompt="sys"), "plain reply")
        url, payload = self.backend._session.calls[-1]
        self.assertTrue(url.endswith("/api/generate"))
        self.assertEqual(payload["prompt"], "hi")
        self.assertEqual(payload["system"], "sys")

    def test_with_history_uses_chat_endpoint_in_order(self):
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                   {"role": "system", "content": "must be dropped"}]
        self.assertEqual(self.backend.generate("c", system_prompt="sys", history=history), "chat reply")
        url, payload = self.backend._session.calls[-1]
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual([(m["role"], m["content"]) for m in payload["messages"]],
                         [("system", "sys"), ("user", "a"), ("assistant", "b"), ("user", "c")])
        self.assertNotIn("prompt", payload)


class _Backend:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append((prompt, system_prompt))
        if self.error:
            raise self.error
        return self.reply


class ComposerTests(unittest.TestCase):
    def test_refusal_is_model_written_from_category_only(self):
        backend = _Backend("I can't help with that, but I can explain defensive basics instead.")
        text, source = composer.compose_refusal(backend, "jailbreak")
        self.assertEqual(source, "model")
        self.assertIn("can't help", text)
        prompt, system = backend.calls[0]
        self.assertTrue(prompt.startswith("Category:"))
        self.assertIn("never the request itself", system)

    def test_unknown_category_still_works(self):
        text, source = composer.compose_refusal(_Backend("Not something I can do."), "something_new")
        self.assertEqual(source, "model")

    def test_falls_back_when_model_down_or_reply_unusable(self):
        for backend in (_Backend(error=ConnectionError("down")), _Backend(""), _Backend("x" * 2000), _Backend(None)):
            text, source = composer.compose_refusal(backend, "harmful", fallback="static text")
            self.assertEqual((text, source), ("static text", "fallback"))

    def test_notice(self):
        text, source = composer.compose_notice(_Backend("Let's try that again."), "answer_withheld")
        self.assertEqual((text, source), ("Let's try that again.", "model"))

    def test_generation_prompt_keeps_leak_marker_sentences(self):
        from core.output_checks import SYSTEM_INSTRUCTION_MARKERS

        lowered = composer.GENERATION_SYSTEM.lower()
        for marker in SYSTEM_INSTRUCTION_MARKERS:
            self.assertIn(marker, lowered)


class AuditorContextTests(unittest.TestCase):
    def test_crisis_support_context_is_added_only_when_requested(self):
        from core.output_auditor import OutputAuditor

        class Backend:
            def __init__(self):
                self.systems = []

            def generate(self, payload, system_prompt=None, **kwargs):
                self.systems.append(system_prompt)
                return '{"safe": true, "relevant": true, "categories": []}'

        backend = Backend()
        auditor = OutputAuditor(backend)
        auditor.review("hi", "Hello there")
        auditor.review("I might hurt myself", "Are you safe right now?", context="crisis_support")
        self.assertNotIn("crisis-support reply", backend.systems[0])
        self.assertIn("crisis-support reply", backend.systems[1])


if __name__ == "__main__":
    unittest.main()

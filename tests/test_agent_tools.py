"""The agent's tools run on the tools MCP server, over the MCP protocol, under policy hooks."""

import unittest
from types import SimpleNamespace

from agent.graph import GuardAgent
from agent.mcp_client import MCPClientError, close_clients
from agent.tool_client import McpToolBackend, ToolsClient
from mcp_server.policy import PolicyViolation
from mcp_server.registry import ServiceNotFound
from mcp_server.services import build_registry
from mcp_server.tools import TOOL_NAMES, build_tool_router


class FakeBackend:
    model = "fake-model"

    def __init__(self, text="A fine answer."):
        self.text, self.calls = text, []

    def generate(self, prompt, system_prompt=None, history=None, schema=None, temperature=None):
        self.calls.append({"prompt": prompt, "system": system_prompt, "history": history,
                           "schema": schema, "temperature": temperature})
        return self.text


class SafeGuard:
    def check(self, *args, **kwargs):
        return SimpleNamespace(status="SAFE", route="release", findings=[],
                               as_dict=lambda: {"status": "SAFE", "route": "release", "findings": []})


def run_agent(agent, route="agent", prompt="What is AI?"):
    return agent.run({"prompt": prompt, "route": route, "intent": "safe", "self_harm_level": "none"})


class RecordingRunner:
    """Stands in for the tools client: records calls, echoes redaction, returns text."""

    def __init__(self, text="via tool", spans=0):
        self.calls, self.text, self.spans = [], text, spans

    def __call__(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "redact_text":
            return {"text": arguments["text"], "pii_types": [], "span_count": self.spans}
        return {"text": self.text}


class AgentUsesToolLayerTests(unittest.TestCase):
    def test_agent_redacts_then_generates_through_the_runner(self):
        backend, runner = FakeBackend(), RecordingRunner()
        state = run_agent(GuardAgent(backend, SafeGuard(), tool_runner=runner))
        self.assertEqual(state["user_message"], "via tool")
        self.assertEqual([name for name, _ in runner.calls], ["redact_text", "ollama_generate"])
        self.assertEqual(backend.calls, [])
        by_tool = {call["tool"]: call for call in state["tool_calls"]}
        self.assertEqual(by_tool["redact_text"]["via"], "mcp")
        self.assertEqual(by_tool["ollama_generate"]["via"], "mcp")

    def test_model_receives_the_redacted_prompt(self):
        runner = RecordingRunner()
        runner_redact = runner.__call__

        def redacting(name, arguments):
            if name == "redact_text":
                runner.calls.append((name, dict(arguments)))
                return {"text": "mail <EMAIL_1>", "pii_types": ["email"], "span_count": 1}
            return runner_redact(name, arguments)

        run_agent(GuardAgent(FakeBackend(), SafeGuard(), tool_runner=redacting), prompt="mail a@b.com")
        self.assertEqual(runner.calls[-1][1]["prompt"], "mail <EMAIL_1>")

    def test_support_route_uses_the_support_tool(self):
        runner = RecordingRunner(text="I am here.")
        run_agent(GuardAgent(FakeBackend(), SafeGuard(), tool_runner=runner), route="support")
        self.assertEqual([name for name, _ in runner.calls], ["redact_text", "ollama_support"])

    def test_without_a_runner_the_backend_is_used_directly(self):
        backend = FakeBackend()
        state = run_agent(GuardAgent(backend, SafeGuard()))
        self.assertEqual(len(backend.calls), 1)
        self.assertNotIn("via", state["tool_calls"][0])

    def test_tool_failure_is_not_released(self):
        def runner(name, arguments):
            raise PolicyViolation("blocked by hook")

        state = run_agent(GuardAgent(FakeBackend(), SafeGuard(), tool_runner=runner))
        self.assertNotEqual(state["outcome"], "released")
        self.assertEqual(state["tool_calls"][0]["status"], "error")


class McpToolBackendTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        client = SimpleNamespace(call=lambda name, args: self.calls.append((name, args)) or {"text": "ok"})
        self.backend = McpToolBackend(client)

    def test_plain_generation_uses_ollama_generate(self):
        self.assertEqual(self.backend.generate("hi", system_prompt="be kind"), "ok")
        self.assertEqual(self.calls[0][0], "ollama_generate")

    def test_schema_call_is_an_output_review(self):
        self.backend.generate("payload", system_prompt="audit", schema={"type": "object"}, temperature=0.0)
        name, arguments = self.calls[0]
        self.assertEqual(name, "ollama_review")
        self.assertEqual(arguments["schema"], {"type": "object"})
        self.assertEqual(arguments["temperature"], 0.0)


class ToolRouterTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.router = build_tool_router(lambda: self.backend)

    def test_registered_tools(self):
        self.assertEqual(set(self.router.registry.names()), set(TOOL_NAMES))
        self.assertTrue({"ollama_generate", "ollama_support", "ollama_review", "redact_text"} <= set(TOOL_NAMES))

    def test_generate_goes_through_the_backend(self):
        result = self.router.dispatch("ollama_generate", {"prompt": "hi", "system": "be kind"})
        self.assertEqual(result, {"text": "A fine answer.", "model": "fake-model"})
        self.assertEqual(self.backend.calls[0]["system"], "be kind")

    def test_review_forwards_schema_and_temperature(self):
        self.router.dispatch("ollama_review", {"prompt": "x", "schema": {"type": "object"}, "temperature": 0.0})
        self.assertEqual(self.backend.calls[0]["schema"], {"type": "object"})
        self.assertEqual(self.backend.calls[0]["temperature"], 0.0)

    def test_redact_text_replaces_pii(self):
        result = self.router.dispatch("redact_text", {"text": "write to jane@example.com please"})
        self.assertNotIn("jane@example.com", result["text"])
        self.assertEqual(result["pii_types"], ["email"])
        self.assertEqual(result["span_count"], 1)

    def test_public_services_are_not_agent_tools(self):
        with self.assertRaises(ServiceNotFound):
            self.router.dispatch("complete_request", {"prompt": "hi"})

    def test_bad_arguments_are_rejected_before_the_model(self):
        for tool, arguments in (
            ("ollama_generate", {"prompt": "x" * 20_000}),
            ("ollama_generate", {"prompt": "  "}),
            ("ollama_review", {"prompt": "x", "schema": "not-an-object"}),
            ("ollama_review", {"prompt": "x", "temperature": 9}),
            ("redact_text", {"text": ""}),
        ):
            with self.subTest(tool=tool, arguments=list(arguments)):
                with self.assertRaises(PolicyViolation):
                    self.router.dispatch(tool, arguments)
        self.assertEqual(self.backend.calls, [])

    def test_empty_model_text_is_rejected(self):
        self.backend.text = "   "
        with self.assertRaises(PolicyViolation):
            self.router.dispatch("ollama_generate", {"prompt": "hi"})

    def test_token_is_required_when_configured(self):
        router = build_tool_router(lambda: self.backend, token="s3cret")
        with self.assertRaises(PolicyViolation):
            router.dispatch("ollama_generate", {"prompt": "hi"})
        with self.assertRaises(PolicyViolation):
            router.dispatch("ollama_generate", {"prompt": "hi", "token": "wrong"})
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(router.dispatch("ollama_generate", {"prompt": "hi", "token": "s3cret"})["text"],
                         "A fine answer.")


class PublicSurfaceTests(unittest.TestCase):
    def test_model_tools_are_not_public_services(self):
        self.assertFalse(set(build_registry().names()) & set(TOOL_NAMES))


class RealToolsServerTests(unittest.TestCase):
    """Starts the real tools MCP server and calls it over the MCP protocol."""

    @classmethod
    def setUpClass(cls):
        cls.client = ToolsClient(startup_timeout=90)
        cls.client.ensure_started()

    @classmethod
    def tearDownClass(cls):
        close_clients()
        cls.client.stop()

    def test_redact_text_over_mcp(self):
        result = self.client.call("redact_text", {"text": "reach me at jane@example.com"})
        self.assertNotIn("jane@example.com", result["text"])
        self.assertEqual(result["pii_types"], ["email"])

    def test_wrong_token_is_rejected_by_the_server(self):
        intruder = ToolsClient(url=self.client.url, token="not-the-token")
        with self.assertRaises(MCPClientError):
            intruder.call("redact_text", {"text": "hello"})

    def test_server_lists_exactly_the_agent_tools(self):
        from agent.mcp_client import list_tools

        self.assertEqual(set(list_tools(url=self.client.url)), set(TOOL_NAMES))


if __name__ == "__main__":
    unittest.main()

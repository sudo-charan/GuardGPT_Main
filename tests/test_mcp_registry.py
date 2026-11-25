"""MCP service registry, routing engine and policy hooks."""

import unittest

from mcp_server.policy import (
    HookContext,
    PolicyHooks,
    PolicyViolation,
    default_hooks,
    validate_session_id,
)
from mcp_server.registry import (
    DuplicateService,
    ServiceNotFound,
    ServiceRegistry,
    ServiceSpec,
)
from mcp_server.router import Router
from mcp_server.services import build_registry


def _spec(name, handler, **kwargs):
    return ServiceSpec(
        name=name, layer="test", description=f"{name} service", handler=handler, **kwargs
    )


class RegistryTests(unittest.TestCase):
    def test_default_registry_matches_the_exposed_tools(self):
        registry = build_registry()
        self.assertEqual(
            set(registry.names()),
            {
                "health",
                "complete_request",
                "session_report",
                "input_guard",
                "intent_analysis",
                "jev_evaluate",
                "policy_decision",
                "review_queue",
            },
        )
        self.assertEqual(len(registry), 8)

    def test_complete_request_is_the_only_canonical_service(self):
        registry = build_registry()
        self.assertEqual(
            [spec.name for spec in registry.services() if spec.canonical],
            ["complete_request"],
        )

    def test_every_architecture_layer_is_hosted(self):
        registry = build_registry()
        self.assertTrue(
            {"layer1", "layer2", "jev", "layer3"} <= {spec.layer for spec in registry.services()}
        )

    def test_duplicate_registration_is_rejected(self):
        registry = ServiceRegistry()
        spec = _spec("echo", lambda arguments: {"ok": True})
        registry.register(spec)
        with self.assertRaises(DuplicateService):
            registry.register(spec)

    def test_unknown_service_is_rejected(self):
        with self.assertRaises(ServiceNotFound):
            ServiceRegistry().get("nope")

    def test_describe_is_json_safe(self):
        described = build_registry().describe()
        self.assertTrue(all("name" in entry and "layer" in entry for entry in described))


class RouterTests(unittest.TestCase):
    def test_dispatch_resolves_the_handler_and_returns_its_mapping(self):
        registry = ServiceRegistry()
        registry.register(_spec("echo", lambda arguments: {"echo": arguments["value"]}))
        self.assertEqual(Router(registry).dispatch("echo", {"value": 7}), {"echo": 7})

    def test_hooks_run_before_and_after_the_handler(self):
        calls = []
        registry = ServiceRegistry()
        registry.register(
            _spec("echo", lambda arguments: calls.append("handler") or {"ok": True})
        )
        hooks = PolicyHooks()
        hooks.register_pre("first", lambda context: calls.append(f"pre:{context.tool}"))
        hooks.register_post("last", lambda context: calls.append(f"post:{context.tool}"))
        Router(registry, hooks).dispatch("echo", {})
        self.assertEqual(calls, ["pre:echo", "handler", "post:echo"])

    def test_pre_hook_prevents_the_handler(self):
        calls = []
        registry = ServiceRegistry()
        registry.register(_spec("danger", lambda arguments: calls.append("handler") or {}))

        def block(context):
            raise PolicyViolation("not allowed")

        hooks = PolicyHooks()
        hooks.register_pre("block", block)
        with self.assertRaises(PolicyViolation):
            Router(registry, hooks).dispatch("danger", {})
        self.assertEqual(calls, [])

    def test_post_hook_can_replace_the_result(self):
        registry = ServiceRegistry()
        registry.register(_spec("echo", lambda arguments: {"ok": True}))
        hooks = PolicyHooks()
        hooks.register_post("stamp", lambda context: dict(context.data, stamped=True))
        self.assertEqual(
            Router(registry, hooks).dispatch("echo", {}), {"ok": True, "stamped": True}
        )

    def test_default_hooks_reject_non_object_results(self):
        registry = ServiceRegistry()
        registry.register(_spec("bad", lambda arguments: "not a mapping"))
        with self.assertRaises(PolicyViolation):
            Router(registry, default_hooks()).dispatch("bad", {})


class SessionPolicyTests(unittest.TestCase):
    def test_session_id_bounds(self):
        validate_session_id(HookContext("t", {}))
        validate_session_id(HookContext("t", {"session_id": "a"}))
        validate_session_id(HookContext("t", {"session_id": "a" * 100}))
        for bad in ("", "a" * 101, 123):
            with self.subTest(session_id=bad):
                with self.assertRaises(PolicyViolation):
                    validate_session_id(HookContext("t", {"session_id": bad}))

    def test_session_policy_blocks_before_the_pipeline_runs(self):
        registry = build_registry()
        with self.assertRaises(PolicyViolation):
            Router(registry, default_hooks()).dispatch(
                "complete_request", {"prompt": "hello", "session_id": "x" * 101}
            )


class HealthServiceTests(unittest.TestCase):
    def test_health_reports_the_registry(self):
        registry = build_registry()
        result = Router(registry, default_hooks()).dispatch("health", {})
        self.assertEqual(result["service"], "GuardGPT")
        self.assertEqual(result["version"], "3.0")
        self.assertTrue(result["complete_pipeline"])
        self.assertEqual(result["registry"]["count"], 8)
        self.assertEqual(
            {entry["name"] for entry in result["registry"]["services"]},
            set(registry.names()),
        )


if __name__ == "__main__":
    unittest.main()

"""Registered services of the GuardGPT MCP server (the architecture backbone).

``complete_request`` is the canonical end-to-end path through every layer.
The per-layer services expose each layer on its own for inspection and
debugging; they run the same objects as the pipeline, so their answers match
what the pipeline decides, but they never generate or release an answer.
"""

from __future__ import annotations

from mcp_server.registry import ServiceRegistry, ServiceSpec

__all__ = ["SERVER_VERSION", "build_registry", "pipeline"]

SERVER_VERSION = "3.0"


def pipeline():
    from core.runtime import complete_pipeline

    pipe = complete_pipeline()
    # Everything that runs a model goes through the tools MCP server (bound once):
    # the agent's tools, the response composer and the output auditor.
    agent = getattr(pipe, "agent", None)
    if agent is not None and getattr(agent, "tool_runner", "unset") is None:
        from agent.tool_client import McpToolBackend, tools_client

        client = tools_client()
        backend = McpToolBackend(client)
        agent.tool_runner = client.call
        agent.backend = backend
        auditor = getattr(getattr(agent, "output_guard", None), "auditor", None)
        if auditor is not None:
            auditor.backend = McpToolBackend(client)
    return pipe


def _prompt(arguments: dict) -> str:
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt


def _complete_request(arguments: dict) -> dict:
    return pipeline().run(
        _prompt(arguments),
        session_id=arguments.get("session_id"),
        check_only=bool(arguments.get("check_only", False)),
        request_id=arguments.get("request_id"),
    )


def _session_report(arguments: dict) -> dict:
    report = pipeline().session_report(arguments.get("session_id"))
    return {"session_report": report}


def _layers(prompt: str):
    """Run Layer 1 -> Layer 2 -> JEV -> Layer 3 without generation or state."""
    from core import runtime
    from core.sensitive_data import redact

    clean = redact(prompt)
    embedding = runtime.intent_layer().embed(clean.text if clean.detected else prompt) if prompt.strip() else None
    guard = runtime.input_guard().check(prompt, embedding=embedding)
    if {"empty_input", "input_too_long"} & set(guard.findings):
        return guard, None, None, None
    intent = runtime.intent_layer().analyze(guard.redacted_prompt, guard.findings, embedding=embedding)
    jev = runtime.jev_model().evaluate(intent, guard, (), prompt=guard.redacted_prompt)
    decision = runtime.policy_engine().decide(guard, intent, jev, guard.redacted_prompt)
    return guard, intent, jev, decision


def _guard_dict(guard) -> dict:
    data = guard.as_dict()
    data.pop("redacted_prompt", None)
    return data


def _input_guard(arguments: dict) -> dict:
    from core import runtime

    return _guard_dict(runtime.input_guard().check(_prompt(arguments)))


def _intent_analysis(arguments: dict) -> dict:
    guard, intent, _, _ = _layers(_prompt(arguments))
    return {"layer1_input": _guard_dict(guard), "layer2_intent": intent.as_dict() if intent else None}


def _jev_evaluate(arguments: dict) -> dict:
    guard, intent, jev, _ = _layers(_prompt(arguments))
    return {"layer2_intent": intent.as_dict() if intent else None, "jev": jev.as_dict() if jev else None}


def _policy_decision(arguments: dict) -> dict:
    guard, intent, jev, decision = _layers(_prompt(arguments))
    return {
        "layer1_input": _guard_dict(guard),
        "layer2_intent": intent.as_dict() if intent else None,
        "jev": jev.as_dict() if jev else None,
        "layer3_policy": decision.as_dict() if decision else None,
    }


def _review_queue(arguments: dict) -> dict:
    from core.review import ReviewQueue

    return {"pending": ReviewQueue().pending()}


def build_registry() -> ServiceRegistry:
    registry = ServiceRegistry()

    def _health(arguments: dict) -> dict:
        from agent.tool_client import tools_client
        from core import runtime
        from core.layers.layer2_intent import validate_checkpoint
        from core.observability import metrics
        from mcp_server.tools import TOOL_NAMES

        checkpoint = validate_checkpoint()
        return {
            "service": "GuardGPT",
            "version": SERVER_VERSION,
            "complete_pipeline": True,
            "intent_model": {"active": bool(checkpoint["ok"]), "reason": checkpoint["reason"],
                             "labels": checkpoint["labels"]},
            "jev_backend": runtime.jev_model().mode,
            "jev": runtime.jev_model().describe(),
            "metrics": metrics().snapshot(),
            "registry": {"count": len(registry), "services": registry.describe()},
            "agent_tools": {**tools_client().describe(), "tools": list(TOOL_NAMES)},
        }

    specs = [
        ("health", "diagnostics", "Server version, model status and registered services.", _health, True, False),
        ("complete_request", "complete",
         "Canonical path: Layer 1 input guard -> Layer 2 intent -> JEV -> Layer 3 policy -> "
         "LangGraph agent / Ollama -> Layer 1 output guard -> audit trail.", _complete_request, True, True),
        ("session_report", "diagnostics", "Session intent history and decayed risk score.", _session_report,
         True, False),
        ("input_guard", "layer1", "Layer 1 input-mode checks only (SAFE / FLAG / UNSAFE).", _input_guard,
         False, False),
        ("intent_analysis", "layer2", "Layer 1 + Layer 2: intent label, confidence and distribution.",
         _intent_analysis, False, False),
        ("jev_evaluate", "jev", "Layer 2 intent + confidence evaluated by the JEV model.", _jev_evaluate,
         False, False),
        ("policy_decision", "layer3", "Layers 1-3 decision (ALLOW / FLAG / BLOCK) without generation.",
         _policy_decision, False, False),
        ("review_queue", "audit", "Pending FLAG reviews (digests only, no prompt text).", _review_queue,
         True, False),
    ]
    for name, layer, description, handler, authoritative, canonical in specs:
        registry.register(ServiceSpec(name=name, layer=layer, description=description, handler=handler,
                                      canonical=canonical, authoritative=authoritative))
    return registry

"""Tool layer: what the LangGraph agent executes, hosted by the tools MCP server.

The agent never talks to Ollama itself. Every execution is an MCP tool call:

* ``ollama_generate`` / ``ollama_support``  answer and crisis-support generation
* ``ollama_review``                         structured output review (the auditor)
* ``redact_text``                           security tool: PII / secret redaction

Each call goes ``name -> tool registry -> pre-hooks -> handler -> post-hooks``,
so the allowlist, argument limits and result checks apply to every tool, and the
router is the same one used by the public services.

Raw generation must never be reachable without Layer 1-3, so the tools server
only accepts calls that carry the per-launch secret token (``GUARDGPT_TOOLS_TOKEN``).
"""

from __future__ import annotations

import hmac
from typing import Any, Callable

from mcp_server.policy import HookContext, PolicyHooks, PolicyViolation, require_mapping_result
from mcp_server.registry import ServiceRegistry, ServiceSpec
from mcp_server.router import Router

__all__ = ["TOOL_NAMES", "build_tool_registry", "build_tool_router"]

TOOL_NAMES = ("ollama_generate", "ollama_support", "ollama_review", "redact_text")
MODEL_TOOLS = frozenset({"ollama_generate", "ollama_support", "ollama_review"})
MAX_PROMPT_CHARS = 12_000
MAX_SYSTEM_CHARS = 8_000
MAX_HISTORY_MESSAGES = 24


def build_tool_registry(
    backend_provider: Callable[[], Any],
    audit_provider: Callable[[], Any] | None = None,
) -> ServiceRegistry:
    """Register the tools. Providers return the Ollama backends (generation, audit)."""
    audit_provider = audit_provider or backend_provider

    def _generate(arguments: dict) -> dict:
        backend = backend_provider()
        history = arguments.get("history")
        if history:
            text = backend.generate(arguments["prompt"], system_prompt=arguments.get("system"), history=history)
        else:
            text = backend.generate(arguments["prompt"], system_prompt=arguments.get("system"))
        return {"text": text, "model": getattr(backend, "model", None)}

    def _review(arguments: dict) -> dict:
        backend = audit_provider()
        text = backend.generate(
            arguments["prompt"],
            system_prompt=arguments.get("system"),
            schema=arguments.get("schema"),
            temperature=arguments.get("temperature"),
        )
        return {"text": text, "model": getattr(backend, "model", None)}

    def _redact(arguments: dict) -> dict:
        from core.sensitive_data import redact

        result = redact(arguments["text"])
        return {"text": result.text, "pii_types": list(result.pii_types), "span_count": result.span_count}

    registry = ServiceRegistry()
    for name, description, handler in (
        ("ollama_generate", "Answer generation with the local Ollama model (after the policy engine).",
         _generate),
        ("ollama_support", "Crisis-support reply generation with the local Ollama model.", _generate),
        ("ollama_review", "Structured output review with the local Ollama audit model.", _review),
        ("redact_text", "Security tool: replace PII and secrets with typed placeholders.", _redact),
    ):
        registry.register(ServiceSpec(name=name, layer="tools", description=description,
                                      handler=handler, authoritative=False))
    return registry


def _authenticated(token: str) -> Callable[[HookContext], None]:
    def hook(context: HookContext) -> None:
        supplied = context.arguments.pop("token", None)
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(), token.encode()):
            raise PolicyViolation("Tool call rejected: missing or invalid tool token")
    return hook


def _allowed_tool(registry: ServiceRegistry) -> Callable[[HookContext], None]:
    def hook(context: HookContext) -> None:
        if context.tool not in TOOL_NAMES or context.tool not in registry:
            raise PolicyViolation(f"Tool not allowed: {context.tool}")
    return hook


def _bounded_arguments(context: HookContext) -> None:
    arguments = context.arguments
    key = "text" if context.tool == "redact_text" else "prompt"
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PolicyViolation(f"Tool argument '{key}' must be a non-empty string")
    if len(value) > MAX_PROMPT_CHARS:
        raise PolicyViolation(f"Tool argument '{key}' is too long")
    if context.tool not in MODEL_TOOLS:
        return
    system = arguments.get("system")
    if system is not None and (not isinstance(system, str) or len(system) > MAX_SYSTEM_CHARS):
        raise PolicyViolation("Tool system prompt is invalid or too long")
    history = arguments.get("history")
    if history is not None and (not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES):
        raise PolicyViolation("Tool history is invalid or too long")
    schema = arguments.get("schema")
    if schema is not None and not isinstance(schema, dict):
        raise PolicyViolation("Tool schema must be an object")
    temperature = arguments.get("temperature")
    if temperature is not None and (not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2):
        raise PolicyViolation("Tool temperature must be between 0 and 2")


def _text_result(context: HookContext) -> None:
    text = context.data.get("text") if isinstance(context.data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise PolicyViolation(f"{context.tool} returned no text")


def build_tool_router(
    backend_provider: Callable[[], Any],
    audit_provider: Callable[[], Any] | None = None,
    token: str | None = None,
) -> Router:
    """Tool router. With ``token`` every call must present it (the tools server always does)."""
    registry = build_tool_registry(backend_provider, audit_provider)
    hooks = PolicyHooks()
    if token:
        hooks.register_pre("authenticated", _authenticated(token))
    hooks.register_pre("allowed_tool", _allowed_tool(registry))
    hooks.register_pre("bounded_arguments", _bounded_arguments)
    hooks.register_post("require_mapping_result", require_mapping_result)
    hooks.register_post("require_text", _text_result)
    return Router(registry, hooks)

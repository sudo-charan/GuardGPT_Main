"""Routing engine for the MCP server.

One place resolves a tool name to its registered handler and applies the policy
hooks in order:

    name -> registry -> pre-hooks -> handler -> post-hooks -> result

Unknown names raise :class:`ServiceNotFound`; a failing pre-hook raises
:class:`PolicyViolation` before the handler is touched.
"""

from __future__ import annotations

from mcp_server.policy import HookContext, PolicyHooks
from mcp_server.registry import ServiceRegistry

__all__ = ["Router"]


class Router:
    def __init__(self, registry: ServiceRegistry, hooks: PolicyHooks | None = None) -> None:
        self.registry = registry
        self.hooks = hooks or PolicyHooks()

    def dispatch(self, name: str, arguments: dict | None = None) -> dict:
        spec = self.registry.get(name)
        context = HookContext(tool=name, arguments=dict(arguments or {}))
        self.hooks.run_pre(context)
        context.data = spec.handler(context.arguments)
        return self.hooks.run_post(context)

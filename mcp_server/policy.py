"""Policy hooks around MCP service calls.

A hook sees the tool name and its arguments before the handler runs, and the
handler's result afterwards. Pre-hooks may reject a call by raising
:class:`PolicyViolation`; post-hooks may replace the result.

Keeping these rules here means the server handlers stay thin and the same
checks apply no matter which transport or wrapper invoked the service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "HookContext",
    "PolicyViolation",
    "PolicyHooks",
    "default_hooks",
    "MAX_SESSION_ID_LENGTH",
]

MAX_SESSION_ID_LENGTH = 100

PreHook = Callable[["HookContext"], None]
PostHook = Callable[["HookContext"], "dict | None"]


class PolicyViolation(ValueError):
    """Raised when a call breaks a server-side policy."""


@dataclass
class HookContext:
    tool: str
    arguments: dict
    data: Any = None
    applied: list[str] = field(default_factory=list)


class PolicyHooks:
    """Ordered pre- and post-hooks applied to every dispatched call."""

    def __init__(self) -> None:
        self._pre: list[tuple[str, PreHook]] = []
        self._post: list[tuple[str, PostHook]] = []

    def register_pre(self, name: str, hook: PreHook) -> None:
        self._pre.append((name, hook))

    def register_post(self, name: str, hook: PostHook) -> None:
        self._post.append((name, hook))

    def run_pre(self, context: HookContext) -> None:
        for name, hook in self._pre:
            hook(context)
            context.applied.append(name)

    def run_post(self, context: HookContext) -> dict:
        for name, hook in self._post:
            updated = hook(context)
            if updated is not None:
                context.data = updated
            context.applied.append(name)
        return context.data

    def describe(self) -> dict:
        return {
            "pre": [name for name, _ in self._pre],
            "post": [name for name, _ in self._post],
        }


def validate_session_id(context: HookContext) -> None:
    """Reject a malformed session id before it reaches the pipeline."""
    session_id = context.arguments.get("session_id")
    if session_id is None:
        return
    if not isinstance(session_id, str) or not 0 < len(session_id) <= MAX_SESSION_ID_LENGTH:
        raise PolicyViolation("Invalid session id")


def require_mapping_result(context: HookContext) -> dict | None:
    """Every service must answer with a JSON object."""
    if not isinstance(context.data, dict):
        raise PolicyViolation(
            f"{context.tool} returned {type(context.data).__name__}, expected an object"
        )
    return None


def default_hooks() -> PolicyHooks:
    hooks = PolicyHooks()
    hooks.register_pre("validate_session_id", validate_session_id)
    hooks.register_post("require_mapping_result", require_mapping_result)
    return hooks

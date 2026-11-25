"""State carried between the LangGraph agent's nodes (Layer 3 orchestration)."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # Inputs set by the pipeline once the policy engine allowed / routed the request.
    trace_id: str
    prompt: str                     # redacted prompt (never raw PII)
    route: str                      # agent | support | refusal | review
    intent: str                     # Layer 2 label
    input_findings: list[str]       # Layer 1 input findings
    self_harm_level: str
    follow_up: str | None
    history: list[dict[str, str]]
    request_span_digests: frozenset
    policy_message: str             # message chosen by the policy engine
    max_attempts: int

    # Working state.
    tool: str                       # tool selected for this step
    system: str
    attempts: int
    candidate: str | None
    output_verdict: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]

    # Result.
    outcome: str                    # released | review | blocked | refused | support | support_fallback
    response: str | None
    user_message: str
    message_source: str             # model | fallback

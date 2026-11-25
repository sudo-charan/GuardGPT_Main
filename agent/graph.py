"""LangGraph agent - Layer 3 orchestration & execution.

Runs inside the MCP server after the Guard / Policy Engine decided. It owns
workflow control, decision routing, tool selection, tool invocation and the
regeneration / re-entry loop:

    START -> route --(agent)----> select_tool -> execute_tool -> output_guard
                 |                      ^                            |
                 |                      +---- regenerate (UNSAFE) ---+
                 |                                                   +-- release (SAFE)  -> END
                 |                                                   +-- review  (FLAG)  -> END
                 |                                                   +-- block   (UNSAFE, no retries left) -> END
                 +--(support)-> select_tool -> execute_tool -> output_guard (crisis context)
                 +--(refusal | review)-> compose_notice -> END

Tools (AI / tool execution box of the architecture):

* ``ollama_generate``  - Ollama / Llama 3.2 answer generation
* ``ollama_support``   - Ollama crisis-support reply
* ``compose_notice``   - short refusal / review notice (model-written, static fallback)

Every tool call is recorded in ``state["tool_calls"]`` and in the audit trail
with the request's trace id. Output from ``execute_tool`` is always routed back
through Layer 1 output mode before it can be released.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

from agent.state import AgentState
from core import output_checks
from core import response_composer as composer
from core.layers.contracts import FLAG, SAFE

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 2
REGENERATE_HINT = (
    " A previous draft failed the output safety check. Give a brief, relevant, "
    "non-actionable answer, or politely decline if the request cannot be answered safely."
)


class GuardAgent:
    """Builds and runs the LangGraph workflow with injected dependencies."""

    def __init__(
        self,
        backend: Any,
        output_guard: Any,
        emit: Callable[..., None] | None = None,
        tool_runner: Callable[[str, dict], dict] | None = None,
    ) -> None:
        self.backend = backend
        self.output_guard = output_guard
        self.emit = emit or (lambda *args, **kwargs: None)
        # When set (the MCP server does this), tools run through the MCP tool
        # router: ``tool_runner(name, arguments) -> {"text": ...}``. Without it the
        # backend is called directly, which keeps the agent usable standalone.
        self.tool_runner = tool_runner
        self._graph = self._build()

    # ------------------------------------------------------------------ nodes
    def route(self, state: AgentState) -> AgentState:
        state.setdefault("attempts", 0)
        state.setdefault("tool_calls", [])
        state.setdefault("max_attempts", DEFAULT_MAX_ATTEMPTS)
        return state

    def select_tool(self, state: AgentState) -> AgentState:
        if state["route"] == "support" and state.get("self_harm_level") == "crisis":
            # Explicit crisis: fixed vetted reply, never model-written.
            state.update(tool="static_crisis", system="")
            return state
        if state["route"] == "support":
            system = composer.CRISIS_SYSTEM
            if state.get("history") or state.get("follow_up") == "carry_over":
                system += composer.CRISIS_CARRY_OVER_ADDENDUM
            state.update(tool="ollama_support", system=system)
            return state
        system = composer.GENERATION_SYSTEM
        level = state.get("self_harm_level", "none")
        if level in {"distress", "passive"} or state.get("follow_up"):
            system += composer.WELLBEING_ADDENDUM
            if level == "passive":
                system += composer.PASSIVE_ADDENDUM
            if state.get("follow_up") == "retraction":
                system += composer.RETRACTION_ADDENDUM
            elif state.get("follow_up") == "carry_over":
                system += composer.CARRY_OVER_ADDENDUM
        if state.get("attempts", 0) > 0:
            system += REGENERATE_HINT
        state.update(tool="ollama_generate", system=system)
        return state

    def execute_tool(self, state: AgentState) -> AgentState:
        state["attempts"] = state.get("attempts", 0) + 1
        history = state.get("history") or []
        call = {"tool": state["tool"], "attempt": state["attempts"],
                "model": getattr(self.backend, "model", None)}
        if state["tool"] == "static_crisis":
            call.update(status="ok", model=None, candidate_length=len(composer.CRISIS_REPLY))
            state["candidate"] = composer.CRISIS_REPLY
            state["tool_calls"].append(call)
            self.emit("tool_call", **call)
            return state
        try:
            if self.tool_runner is not None:
                # Security tool first: nothing sensitive reaches the model, even if an
                # earlier stage missed it. The prompt is normally already redacted.
                screened = self.tool_runner("redact_text", {"text": state["prompt"]})
                redact_call = {"tool": "redact_text", "via": "mcp", "status": "ok",
                               "span_count": int(screened.get("span_count", 0))}
                state["tool_calls"].append(redact_call)
                self.emit("tool_call", **redact_call)
                arguments = {"prompt": screened["text"], "system": state["system"]}
                if history:
                    arguments["history"] = history
                candidate = self.tool_runner(state["tool"], arguments)["text"]
                call["via"] = "mcp"
            elif history:
                candidate = self.backend.generate(state["prompt"], system_prompt=state["system"], history=history)
            else:
                candidate = self.backend.generate(state["prompt"], system_prompt=state["system"])
            if not isinstance(candidate, str) or not candidate.strip() or len(candidate) > 16000:
                raise ValueError("invalid or oversized model response")
            call.update(status="ok", candidate_length=len(candidate),
                        candidate_sha256=hashlib.sha256(candidate.encode("utf-8")).hexdigest())
            state["candidate"] = candidate
        except Exception as error:
            call.update(status="error", error=type(error).__name__)
            state["candidate"] = None
        state["tool_calls"].append(call)
        self.emit("tool_call", **call)
        return state

    def output_guard_node(self, state: AgentState) -> AgentState:
        candidate = state.get("candidate")
        if candidate is None:
            state["output_verdict"] = {"status": "ERROR", "route": "tool_error", "findings": ["tool_failed"]}
            return state
        if state.get("tool") == "static_crisis":
            # Fixed, pre-vetted text: nothing to review.
            state["output_verdict"] = {"status": SAFE, "route": "release", "findings": []}
            return state
        # Cosmetic markdown slips (an unclosed code fence) must not withhold a harmless answer.
        candidate, repairs = output_checks.repair_markdown(candidate)
        state["candidate"] = candidate
        verdict = self.output_guard.check(
            state["prompt"], candidate,
            intent=state.get("intent", "safe"),
            input_findings=state.get("input_findings", ()),
            request_span_digests=state.get("request_span_digests", frozenset()),
            context="crisis_support" if state["route"] == "support" else None,
        )
        state["output_verdict"] = verdict.as_dict()
        self.emit("output_guard", status=verdict.status, route=verdict.route,
                  findings=verdict.findings, attempt=state["attempts"], repairs=repairs)
        return state

    def finish_release(self, state: AgentState) -> AgentState:
        outcome = "support" if state["route"] == "support" else "released"
        state.update(outcome=outcome, response=state["candidate"], user_message=state["candidate"],
                     message_source="fixed" if state.get("tool") == "static_crisis" else "model")
        return state

    def finish_review(self, state: AgentState) -> AgentState:
        text, source = composer.compose_notice(self.backend, "answer_withheld",
                                               fallback=composer.STATIC_NOTICES["answer_withheld"])
        state.update(outcome="review", response=None, user_message=text, message_source=source)
        return state

    def finish_block(self, state: AgentState) -> AgentState:
        if state["route"] == "support":
            # A person at risk always gets a reply.
            state.update(outcome="support_fallback", response=composer.SUPPORT_FALLBACK,
                         user_message=composer.SUPPORT_FALLBACK, message_source="fallback")
            return state
        situation = "answer_unsafe"
        text, source = composer.compose_notice(self.backend, situation,
                                               fallback=composer.STATIC_NOTICES[situation])
        failed_tool = (state.get("output_verdict") or {}).get("route") == "tool_error"
        state.update(outcome="error" if failed_tool else "blocked", response=None,
                     user_message=text, message_source=source)
        return state

    def compose_notice(self, state: AgentState) -> AgentState:
        state["tool"] = "compose_notice"
        intent = state.get("intent", "policy")
        if state["route"] == "refusal":
            text, source = composer.compose_refusal(self.backend, intent, fallback=state.get("policy_message"))
            outcome = "refused"
        else:
            text, source = state.get("policy_message") or composer.STATIC_NOTICES["answer_withheld"], "policy"
            outcome = "review"
        state.setdefault("tool_calls", []).append({"tool": "compose_notice", "source": source})
        self.emit("tool_call", tool="compose_notice", source=source)
        state.update(outcome=outcome, response=None, user_message=text, message_source=source)
        return state

    # ---------------------------------------------------------------- routing
    @staticmethod
    def _after_route(state: AgentState) -> str:
        return "select_tool" if state["route"] in {"agent", "support"} else "compose_notice"

    @staticmethod
    def _after_output(state: AgentState) -> str:
        verdict = state.get("output_verdict") or {}
        status = verdict.get("status")
        if status == SAFE:
            return "release"
        if status == FLAG:
            # Support replies are never withheld: a soft finding still releases
            # only if the reviewer did not object; otherwise use the fallback.
            return "block" if state["route"] == "support" else "review"
        # UNSAFE or tool error: regenerate while attempts remain (re-entry).
        if state.get("attempts", 0) < state.get("max_attempts", DEFAULT_MAX_ATTEMPTS):
            return "regenerate"
        return "block"

    # ------------------------------------------------------------------ graph
    def _build(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception as error:  # pragma: no cover - langgraph is a requirement
            logger.warning("LangGraph unavailable (%s); using the sequential runner", error)
            return None
        graph = StateGraph(AgentState)
        graph.add_node("route", self.route)
        graph.add_node("select_tool", self.select_tool)
        graph.add_node("execute_tool", self.execute_tool)
        graph.add_node("output_guard", self.output_guard_node)
        graph.add_node("release", self.finish_release)
        graph.add_node("review", self.finish_review)
        graph.add_node("block", self.finish_block)
        graph.add_node("compose_notice", self.compose_notice)
        graph.set_entry_point("route")
        graph.add_conditional_edges("route", self._after_route,
                                    {"select_tool": "select_tool", "compose_notice": "compose_notice"})
        graph.add_edge("select_tool", "execute_tool")
        graph.add_edge("execute_tool", "output_guard")
        graph.add_conditional_edges("output_guard", self._after_output,
                                    {"release": "release", "review": "review",
                                     "block": "block", "regenerate": "select_tool"})
        for node in ("release", "review", "block", "compose_notice"):
            graph.add_edge(node, END)
        return graph.compile()

    @property
    def uses_langgraph(self) -> bool:
        return self._graph is not None

    def run(self, state: AgentState) -> AgentState:
        if self._graph is not None:
            return self._graph.invoke(state, {"recursion_limit": 25})
        # Sequential fallback with identical semantics.
        state = self.route(state)
        if self._after_route(state) == "compose_notice":
            return self.compose_notice(state)
        while True:
            state = self.output_guard_node(self.execute_tool(self.select_tool(state)))
            step = self._after_output(state)
            if step == "regenerate":
                continue
            return {"release": self.finish_release, "review": self.finish_review,
                    "block": self.finish_block}[step](state)


__all__ = ["GuardAgent"]

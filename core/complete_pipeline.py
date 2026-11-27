"""One request through the GuardGPT architecture.

    User prompt
      -> Layer 1 Guard, input mode       (SAFE | FLAG | UNSAFE)
      -> Layer 2 Intent Intelligence     (6 labels + confidence)
      -> JEV model                       (intent + confidence -> JEV result)
      -> Layer 3 Guard / Policy Engine   (ALLOW | FLAG | BLOCK)
      -> LangGraph agent + Ollama tools  (ALLOW, and support replies)
      -> Layer 1 Guard, output mode      (SAFE release | FLAG review | UNSAFE block/regenerate)
      -> safe response to the host

Cross-cutting: every stage writes an audit-trail event with the request's
trace id (``logs/guardgpt_trace.jsonl``), plus the per-request reports,
compact audit log, request log, error log, review queue and metrics.

No generated candidate is released unless Layer 1 output mode passed it.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from core import escalation
from core.layers.contracts import ALLOW, BLOCK, FLAG
from core.layers.jev import JEVError
from core.observability import (
    append_jsonl,
    audit_trail,
    error_log,
    logs_dir,
    metrics,
    new_trace,
    request_log,
)
from core.review import ReviewQueue
from core.sensitive_data import redact
from core.session_memory import SessionMemory

logger = logging.getLogger(__name__)

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
MAX_SESSIONS = 100

# What the user is told when an input is flagged (the reviewer and priority are in the review record).
REVIEW_MESSAGE = (
    "I can't answer this one right now. It has been sent for a safety review. "
    "If it was a harmless question, try rephrasing it with a bit more context and I'll look again."
)
ESCALATED_MESSAGE = (
    "I can't answer this one right now. Because of the risk level or earlier flagged messages in this "
    "chat, it has been escalated to a human reviewer. If it was a harmless question, try rephrasing it "
    "with more context."
)

STATUS_FOR_OUTCOME = {
    "released": (ALLOW, "SAFE"),
    "support": (BLOCK, "SUPPORT"),
    "support_fallback": (BLOCK, "SUPPORT"),
    "review": (FLAG, "REVIEW"),
    "blocked": (BLOCK, "UNSAFE"),
    "refused": (BLOCK, "UNSAFE"),
    "error": (BLOCK, "ERROR"),
}


def _carried_only_block(guard, jev, decision) -> bool:
    """True when the carried-risk rule alone turned this turn into a BLOCK.

    The rule only upgrades a JEV FLAG, so without a Layer 1 UNSAFE verdict the
    block came from carried risk and must not raise the carried risk again.
    """
    return (decision.action == BLOCK and "escalated_carried_risk" in jev.rationale
            and guard.status != "UNSAFE")


class AuditLog:
    """Compact security-event log: identifiers and verdicts, no prompt text."""

    FIELDS = (
        "timestamp", "trace_id", "request_id", "audit_id", "session_id", "turn_index",
        "layer1_status", "topic", "intent", "intent_confidence", "jev_verdict", "policy_action",
        "risk_level", "action", "final_status", "allowed", "reason_codes", "pii_types",
        "redacted", "review_id",
    )

    def __init__(self, path=None):
        self.path = Path(path or logs_dir() / "guardgpt_audit.jsonl")

    def write(self, event):
        entry = {key: event[key] for key in self.FIELDS
                 if key in event and (event[key] is not None or key not in {"session_id", "turn_index"})}
        append_jsonl(self.path, entry)


class ReportWriter:
    """Detailed prompt / session / complete reports, one JSONL file each."""

    def __init__(self, directory):
        directory = Path(directory)
        self.prompt_path = directory / "guardgpt_prompt.jsonl"
        self.session_path = directory / "guardgpt_session.jsonl"
        self.complete_path = directory / "guardgpt_complete.jsonl"

    def write(self, prompt_report, session_report, complete_report):
        append_jsonl(self.prompt_path, prompt_report)
        if session_report is not None:
            append_jsonl(self.session_path, session_report)
        append_jsonl(self.complete_path, complete_report)


class CompletePipeline:
    def __init__(
        self,
        input_guard=None,
        intent_layer=None,
        jev=None,
        policy=None,
        agent=None,
        backend=None,
        auditor=None,
        audit_log=None,
        review_queue=None,
        max_attempts: int = 2,
        topic_layer=None,
        context_layer=None,
    ):
        from core import runtime

        self.input_guard = input_guard or runtime.input_guard()
        self.intent_layer = intent_layer or runtime.intent_layer()
        self.jev = jev or runtime.jev_model()
        self.policy = policy or runtime.policy_engine()
        if agent is None:
            import os

            from agent.graph import GuardAgent
            from core.layers.layer1_guard import OutputGuard
            from core.llama_backend import LlamaBackend
            from core.output_auditor import OutputAuditor

            backend = backend or LlamaBackend()
            if auditor is None:
                auditor = OutputAuditor(LlamaBackend(model=os.getenv("OLLAMA_AUDIT_MODEL") or backend.model))
            agent = GuardAgent(backend, OutputGuard(auditor))
        self.agent = agent
        # Topic of the message (coding, greeting ...): informational, never affects a decision.
        self.topic_layer = topic_layer
        if self.topic_layer is None and callable(getattr(self.intent_layer, "_encode", None)):
            from core.layers.topic_intent import TopicIntent

            self.topic_layer = TopicIntent(self.intent_layer._encode)
        # Task + conversation context (Layer 2 context understanding): informational too.
        self.context_layer = context_layer
        if self.context_layer is None and callable(getattr(self.intent_layer, "_encode", None)):
            from core.layers.context_understanding import ContextAnalyzer

            self.context_layer = ContextAnalyzer(self.intent_layer._encode)
        self.max_attempts = max(1, min(int(max_attempts), 3))
        self.audit_log = audit_log or AuditLog()
        directory = Path(self.audit_log.path).parent
        self.report_writer = ReportWriter(directory)
        self.trace_path = directory / "guardgpt_trace.jsonl"
        self.escalation_path = directory / "guardgpt_escalations.jsonl"
        self.review_queue = review_queue or ReviewQueue(directory / "guardgpt_review.jsonl")
        self.memories: "OrderedDict[str, SessionMemory]" = OrderedDict()
        self.lock = threading.RLock()
        self._session_locks: "OrderedDict[str, threading.RLock]" = OrderedDict()
        self._session_locks_guard = threading.Lock()

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _request_id(candidate):
        if isinstance(candidate, str) and REQUEST_ID_RE.match(candidate.strip()):
            return candidate.strip()
        return "req_" + uuid.uuid4().hex[:12]

    def _lock_for(self, session_id):
        key = session_id or "__stateless__"
        with self._session_locks_guard:
            lock = self._session_locks.get(key)
            if lock is None:
                while len(self._session_locks) >= 256:
                    for old_key, old_lock in list(self._session_locks.items()):
                        if old_lock.acquire(blocking=False):
                            old_lock.release()
                            del self._session_locks[old_key]
                            break
                    else:
                        break
                lock = self._session_locks[key] = threading.RLock()
            else:
                self._session_locks.move_to_end(key)
            return lock

    def _memory_for(self, session_id):
        if not session_id:
            return None
        with self.lock:
            memory = self.memories.get(session_id)
            if memory is None:
                while len(self.memories) >= MAX_SESSIONS:
                    self.memories.popitem(last=False)
                memory = self.memories[session_id] = SessionMemory()
            self.memories.move_to_end(session_id)
            return memory

    def session_report(self, session_id, request_id="session_query"):
        memory = self.memories.get(session_id) if session_id else None
        return memory.report(session_id, request_id) if memory is not None else None

    # --------------------------------------------------------------------- run
    def run(self, prompt, session_id=None, check_only=False, request_id=None):
        with self._lock_for(session_id):
            return self._run(prompt, session_id, check_only, request_id)

    def _run(self, prompt, session_id, check_only, request_id=None):
        request_id = self._request_id(request_id)
        trace_id = uuid.uuid4().hex
        report = dict(
            trace_id=trace_id, request_id=request_id, audit_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(), session_id=session_id,
            # Redacted from the start, so a prompt that never passes Layer 1 (over-long input,
            # a Layer 1 failure) is not logged with its personal data.
            prompt=redact(prompt).text if isinstance(prompt, str) else "", mode="check" if check_only else "answer",
            action=BLOCK, final_status="ERROR", allowed=False, response=None, user_message="",
            intent="unknown", intent_confidence=0.0, intent_probabilities={}, risk_level="unknown",
            reasons=[], pii_types=[], pii_span_count=0, redacted=False,
            layer1_input=None, layer2_intent=None, jev=None, layer3_policy=None,
            agent=None, layer1_output=None, review=None, turn_index=None,
        )
        trace = new_trace(request_id)

        def emit(event, **fields):
            audit_trail(event, trace_id, path=self.trace_path, request_id=request_id,
                        session_id=session_id, **fields)

        emit("request", mode=report["mode"])
        try:
            self._process(report, prompt, session_id, check_only, trace, emit)
        except JEVError as error:
            report.update(action=BLOCK, final_status="ERROR", allowed=False, response=None,
                          error=type(error).__name__, error_detail=str(error)[:300],
                          user_message="The JEV model is unavailable, so this request was blocked. "
                                       + str(error)[:300])
            report["reasons"].append("jev_failed")
            metrics().incr("jev_failures")
            error_log({"trace_id": trace_id, "request_id": request_id, "stage": "jev",
                       "error": type(error).__name__})
            emit("error", stage="jev", error=type(error).__name__, detail=str(error)[:300])
        except Exception as error:
            logger.exception("Pipeline failure")
            report.update(action=BLOCK, final_status="ERROR", allowed=False, response=None,
                          error=type(error).__name__,
                          user_message="A required component failed. Run python main.py --status.")
            report["reasons"].append("processing_failed")
            metrics().incr("processing_failures")
            error_log({"trace_id": trace_id, "request_id": request_id, "stage": "pipeline",
                       "error": type(error).__name__})
            emit("error", stage="pipeline", error=type(error).__name__)

        memory = self.memories.get(session_id) if session_id else None
        session_report = memory.report(session_id, request_id) if memory is not None else None
        timing = trace.as_dict()
        report["timings_ms"], report["duration_ms"] = timing["stages_ms"], timing["total_ms"]
        try:
            metrics().record_request(report["action"], report["final_status"], trace)
        except Exception:
            pass
        request_log({
            "trace_id": trace_id, "request_id": request_id, "audit_id": report["audit_id"],
            "session_id": session_id, "action": report["action"], "final_status": report["final_status"],
            "intent": report["intent"], "risk_level": report["risk_level"], "mode": report["mode"],
            "stages_ms": timing["stages_ms"], "total_ms": timing["total_ms"],
        })

        final_output = dict(report)
        final_output["session_report"] = session_report
        try:
            # A BLOCK caused only by carried risk is also queued, so a wrong block can be spotted.
            carried_block = (report["action"] == BLOCK and report["final_status"] == "UNSAFE"
                             and "jev:escalated_carried_risk" in report["reasons"])
            # An explicit crisis still gets its support reply at once, and a human is told as well.
            crisis = (report["final_status"] == "SUPPORT"
                      and "self_harm_crisis" in ((report.get("layer1_input") or {}).get("findings") or []))
            if (report["action"] == FLAG and report["final_status"] == "REVIEW") or carried_block or crisis:
                record = self._review_record(report)
                record.update(escalation.assess(
                    record, memory.recent_turns(escalation.REPEAT_WINDOW + 1) if memory is not None else (),
                    memory.session_risk() if memory is not None else 0.0))
                if crisis:
                    record.update(stage="support", priority="urgent", escalated=True,
                                  escalation_reasons=list(dict.fromkeys(
                                      ["crisis_support"] + record.get("escalation_reasons", []))))
                report["review"] = self.review_queue.enqueue(record)
                emit("review_queued", review_id=report["review"].get("review_id"),
                     priority=record["priority"], escalated=record["escalated"])
                if record["escalated"]:
                    self._escalate(report["review"], emit)
                # Input-stage flag: tell the user clearly what happens next.
                if (report.get("layer3_policy") or {}).get("route") == "review":
                    message = (ESCALATED_MESSAGE if record["escalated"] else REVIEW_MESSAGE)
                    report["user_message"] = final_output["user_message"] = message
            prompt_document = self._prompt_report(report)
            session_document = self._session_document(session_report, report) if session_report else None
            self.report_writer.write(prompt_document, session_document,
                                     self._complete_report(report, prompt_document, session_document))
            self.audit_log.write(self._audit_event(report))
            final_output["audit_logged"] = True
        except Exception:
            final_output.update(action=BLOCK, final_status="ERROR", allowed=False, response=None,
                                audit_logged=False,
                                user_message="The audit log could not be saved; no answer was released.")
            final_output["reasons"] = report["reasons"] + ["audit_log_failed"]
            metrics().incr("audit_failures")
            error_log({"trace_id": trace_id, "request_id": request_id, "stage": "audit_write",
                       "error": "audit_log_failed"})
        final_output["review"] = report.get("review")
        final_output["prompt_report"] = self._prompt_report(report)
        emit("response", action=final_output["action"], final_status=final_output["final_status"],
             allowed=final_output["allowed"])
        return final_output

    def _process(self, report, prompt, session_id, check_only, trace, emit):
        stage = trace.stage if trace is not None else (lambda name: nullcontext())
        memory = self._memory_for(session_id)

        # ---------------- Layer 1 - input mode ----------------
        with stage("layer1_input"):
            embedding = None
            if isinstance(prompt, str) and prompt.strip():
                clean = redact(prompt)
                embedding = self.intent_layer.embed(clean.text if clean.detected else prompt)
            guard = self.input_guard.check(prompt, memory.levels if memory is not None else (), embedding=embedding)
        report["layer1_input"] = guard.as_dict()
        report["layer1_input"].pop("redacted_prompt", None)
        emit("security_event", layer="layer1_input", status=guard.status, risk_score=guard.risk_score,
             findings=guard.findings)
        if {"empty_input", "input_too_long"} & set(guard.findings):
            report.update(action=BLOCK, final_status="INVALID_INPUT", allowed=False, risk_level="safe",
                          user_message="Please enter a prompt (maximum 6,000 UTF-8 bytes).",
                          reasons=list(guard.findings))
            return
        report.update(prompt=guard.redacted_prompt, pii_types=guard.pii_types,
                      pii_span_count=guard.pii_span_count, redacted=bool(guard.pii_types))
        span_digests = frozenset(span.digest for span in redact(prompt).spans)

        # ---------------- Layer 2 - intent intelligence ----------------
        with stage("layer2_intent"):
            intent = self.intent_layer.analyze(guard.redacted_prompt, guard.findings, embedding=embedding)
        report["layer2_intent"] = intent.as_dict()
        report.update(intent=intent.intent, intent_confidence=intent.confidence,
                      intent_probabilities=intent.probabilities)
        emit("intent", intent=intent.intent, confidence=intent.confidence,
             passed_threshold=intent.passed_threshold, probabilities=intent.probabilities,
             sources=sorted(k for k in intent.sources if k in {"head", "anchors", "dataset", "guard"}))

        topic = None
        if self.topic_layer is not None:
            try:
                topic = self.topic_layer.classify(embedding)
            except Exception:  # informational only
                topic = None
        report["topic"] = topic

        context = None
        if self.context_layer is not None:
            try:
                context = self.context_layer.analyze(
                    guard.redacted_prompt, embedding, (topic or {}).get("topic"), memory)
            except Exception:  # informational only
                context = None
        report["layer2_intent"]["context"] = context

        # ---------------- JEV model ----------------
        with stage("jev"):
            carried_risk = memory.carried_risk() if memory is not None else 0.0
            jev = self.jev.evaluate(intent, guard, memory.recent_turns() if memory is not None else (),
                                    prompt=guard.redacted_prompt, carried_risk=carried_risk)
        report["jev"] = jev.as_dict()
        report["jev"]["carried_risk"] = carried_risk
        emit("jev", verdict=jev.verdict, label=jev.label, confidence=jev.confidence,
             risk_score=jev.risk_score, rationale=jev.rationale, backend=jev.backend,
             carried_risk=carried_risk)

        # ---------------- Layer 3 - guard / policy engine ----------------
        with stage("layer3_policy"):
            decision = self.policy.decide(guard, intent, jev, guard.redacted_prompt)
        report["layer3_policy"] = decision.as_dict()
        report.update(risk_level=decision.risk_level, reasons=list(decision.reason_codes))
        emit("policy_decision", action=decision.action, route=decision.route,
             risk_level=decision.risk_level, reason_codes=decision.reason_codes)
        if memory is not None:
            memory.record_level("crisis" if decision.route == "support" else guard.self_harm_level)
            report["turn_index"] = memory.record_turn(
                intent=intent.intent, confidence=intent.confidence,
                action=decision.action, risk_score=decision.risk_score,
                topic=(topic or {}).get("topic"), embedding=embedding,
                support=decision.route == "support",
                carried_escalation=_carried_only_block(guard, jev, decision))

        if check_only:
            message = decision.user_message or "Input checks complete; no answer generated."
            if decision.route == "support":
                action, status = BLOCK, "SUPPORT"
                message = "Self-harm risk detected. A supportive reply is written when answering is enabled."
            else:
                action, status = decision.action, {ALLOW: "SAFE", FLAG: "REVIEW", BLOCK: "UNSAFE"}[decision.action]
            report.update(action=action, final_status=status, allowed=action == ALLOW, user_message=message)
            return

        # ---------------- LangGraph agent -> tools -> Layer 1 output ----------------
        state = {
            "trace_id": report["trace_id"],
            "prompt": guard.redacted_prompt,
            "route": decision.route,
            "intent": intent.intent,
            "input_findings": list(guard.findings),
            "self_harm_level": ("distress" if "wellbeing_mode" in decision.reason_codes
                                and guard.self_harm_level == "none" else guard.self_harm_level),
            "follow_up": (guard.checks.get("self_harm") or {}).get("follow_up"),
            "history": memory.messages() if memory is not None else [],
            "request_span_digests": span_digests,
            "policy_message": decision.user_message,
            "max_attempts": self.max_attempts,
        }
        if hasattr(self.agent, "emit"):
            self.agent.emit = emit
        with stage("agent"):
            result = self.agent.run(state)
        outcome = result.get("outcome", "error")
        report["agent"] = {
            "route": decision.route,
            "outcome": outcome,
            "attempts": result.get("attempts", 0),
            "tool_calls": result.get("tool_calls", []),
            "message_source": result.get("message_source"),
            "langgraph": bool(getattr(self.agent, "uses_langgraph", False)),
        }
        report["layer1_output"] = result.get("output_verdict")
        action, status = STATUS_FOR_OUTCOME.get(outcome, (BLOCK, "ERROR"))
        if decision.route == "review":
            action, status = FLAG, "REVIEW"
        report.update(action=action, final_status=status, allowed=action == ALLOW,
                      response=result.get("response"), user_message=result.get("user_message", ""))
        if outcome in {"review", "blocked"} and decision.route == "agent":
            report["reasons"].append("output_" + ("flagged" if outcome == "review" else "blocked"))
        if outcome == "support_fallback":
            report["reasons"].append("support_fallback_used")
        if memory is not None and result.get("response"):
            memory.add_exchange(guard.redacted_prompt, result["response"])

    def _escalate(self, review, emit):
        """Record an escalated review: escalation log, metrics (urgent raises an alert), trace event."""
        entry = {key: review.get(key) for key in (
            "review_id", "created_at", "trace_id", "request_id", "session_id", "turn_index", "priority",
            "escalation_reasons", "intent", "risk_level", "status")}
        try:
            append_jsonl(self.escalation_path, entry)
        except Exception:  # escalation logging must never break a request
            error_log({"trace_id": review.get("trace_id"), "stage": "escalation_write",
                       "error": "escalation_log_failed"})
        metrics().incr("escalations")
        if review.get("priority") == "urgent":
            metrics().incr("escalations_urgent")
        emit("escalation", review_id=review.get("review_id"), priority=review.get("priority"),
             reasons=review.get("escalation_reasons"))

    # ----------------------------------------------------------------- reports
    @staticmethod
    def _prompt_report(report):
        return {
            "report_type": "prompt",
            "trace_id": report["trace_id"],
            "request_id": report["request_id"],
            "audit_id": report["audit_id"],
            "timestamp": report["timestamp"],
            "session_id": report.get("session_id"),
            "turn_index": report.get("turn_index"),
            "prompt": {"text": report.get("prompt", ""), "mode": report.get("mode")},
            "layer1_input": report.get("layer1_input"),
            "topic": report.get("topic"),
            "layer2_intent": report.get("layer2_intent"),
            "jev": report.get("jev"),
            "layer3_policy": report.get("layer3_policy"),
            "agent": report.get("agent"),
            "layer1_output": report.get("layer1_output"),
            "review": report.get("review"),
            "final_result": {
                "action": report.get("action"),
                "status": report.get("final_status"),
                "allowed": report.get("allowed"),
                "reasons": report.get("reasons", []),
            },
        }

    @staticmethod
    def _session_document(session_report, report):
        return {"report_type": "session", "trace_id": report["trace_id"],
                "audit_id": report["audit_id"], "generated_at": report["timestamp"], **session_report}

    @staticmethod
    def _complete_report(report, prompt_report, session_report):
        return {
            "report_type": "complete",
            "trace_id": report["trace_id"],
            "request_id": report["request_id"],
            "audit_id": report["audit_id"],
            "timestamp": report["timestamp"],
            "session_id": report.get("session_id"),
            "prompt_report": prompt_report,
            "session_report": session_report,
            "execution": {"final_action": report.get("action"), "final_status": report.get("final_status"),
                          "allowed": report.get("allowed")},
        }

    @staticmethod
    def _audit_event(report):
        review = report.get("review") or {}
        return {
            "timestamp": report["timestamp"],
            "trace_id": report["trace_id"],
            "request_id": report["request_id"],
            "audit_id": report["audit_id"],
            "session_id": report.get("session_id"),
            "turn_index": report.get("turn_index"),
            "layer1_status": (report.get("layer1_input") or {}).get("status"),
            "topic": (report.get("topic") or {}).get("topic"),
            "intent": report.get("intent"),
            "intent_confidence": report.get("intent_confidence"),
            "jev_verdict": (report.get("jev") or {}).get("verdict"),
            "policy_action": (report.get("layer3_policy") or {}).get("action"),
            "risk_level": report.get("risk_level"),
            "action": report.get("action"),
            "final_status": report.get("final_status"),
            "allowed": report.get("allowed"),
            "reason_codes": report.get("reasons", []),
            "pii_types": report.get("pii_types", []),
            "redacted": bool(report.get("redacted")),
            "review_id": review.get("review_id"),
        }

    @staticmethod
    def _review_record(report):
        output = report.get("layer1_output") or {}
        calls = (report.get("agent") or {}).get("tool_calls") or []
        last = calls[-1] if calls else {}
        return {
            "trace_id": report["trace_id"],
            "request_id": report["request_id"],
            "audit_id": report["audit_id"],
            "session_id": report.get("session_id"),
            "turn_index": report.get("turn_index"),
            "stage": "output" if output else "input",
            "intent": report.get("intent"),
            "intent_confidence": report.get("intent_confidence"),
            "risk_level": report.get("risk_level"),
            "final_status": report.get("final_status"),
            "reason_codes": list(report.get("reasons", [])),
            "failed_checks": list(output.get("findings") or []),
            "candidate_sha256": last.get("candidate_sha256"),
            "candidate_length": last.get("candidate_length"),
            "prompt_sha256": hashlib.sha256(str(report.get("prompt", "")).encode("utf-8")).hexdigest(),
        }

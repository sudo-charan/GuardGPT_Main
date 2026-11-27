"""JEV model - the step between Layer 2 and Layer 3.

    Input : intent + confidence (the Layer 2 IntentResult, including its full
            probability distribution), the Layer 1 input verdict, and the
            session's recent turns and its carried risk (earlier blocked turns,
            fading over time). In ``finetuned`` mode the prompt text is an
            additional input, because the fine-tuned model reads it.
    Output: JEV result (verdict ALLOW | FLAG | BLOCK, risk score, rationale)
            -> Layer 3 Guard / Policy Engine.

In ``builtin`` mode the JEV never re-classifies the prompt: it evaluates exactly
the intent and confidence Layer 2 produced (the old implementation called the
classifier a second time and collapsed every non-attack label into a single
"Safe/Educational" choice). In ``finetuned`` mode the model classifies the prompt
itself, and that verdict is combined with the intent + confidence evaluation, which
stays as a floor.

Backends (``JEV_STYLE_MODE``):

* ``builtin`` (default) - the deterministic evaluator below; nothing to start.
* ``finetuned`` - the team's fine-tuned Qwen3.5 JEV checkpoint, loaded from a
  local path (see :mod:`core.layers.jev_finetuned`). Its six-class verdict is
  combined with the Layer 2 intent + confidence; the builtin evaluation is a
  floor, so the model can only make the verdict stricter. A missing or broken
  checkpoint raises ``JEVError`` (request blocked, never silently safe).
* ``server``  - an external Jev-Style HTTP server (``JEV_STYLE_URL``). Its
  answer is mapped onto the same JEVResult. Failure raises ``JEVError``.
* ``auto``    - try the server, fall back to the builtin evaluator.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from core.layers.contracts import (
    ALLOW,
    BLOCK,
    FLAG,
    UNSAFE,
    GuardVerdict,
    IntentResult,
    JEVResult,
    UNSAFE_INTENTS,
    risk_from_score,
)
from core.layers.layer1_guard import FINDING_LABEL
from core.layers.layer2_intent import INTENT_BASE_RISK

MODEL_NAME = "guardgpt-jev-builtin-v2"


class JEVError(RuntimeError):
    """The configured JEV backend could not produce a verdict."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Server label -> architecture intent label.
SERVER_LABELS = {
    "safe/educational": "safe",
    "safe": "safe",
    "educational": "safe",
    "prompt injection": "prompt_injection",
    "jailbreak": "jailbreak",
    "harmful instructions": "harmful_instructions",
    "cyber abuse": "harmful_instructions",
    "manipulation": "manipulation",
    "self-harm risk": "self_harm_risk",
    "self_harm_risk": "self_harm_risk",
}


class JEVModel:
    def __init__(
        self,
        block_confidence: float | None = None,
        flag_unsafe_mass: float | None = None,
        mode: str | None = None,
        server_client: Any = None,
        finetuned: Any = None,
    ) -> None:
        # An uncorroborated unsafe label needs this much confidence to BLOCK.
        self.block_confidence = (
            block_confidence if block_confidence is not None
            else _env_float("JEV_BLOCK_THRESHOLD", 0.75)
        )
        # Total unsafe probability at/above which an unsafe label that missed
        # its threshold is still flagged rather than allowed.
        self.flag_unsafe_mass = (
            flag_unsafe_mass if flag_unsafe_mass is not None
            else _env_float("JEV_FLAG_THRESHOLD", 0.50)
        )
        # A "safe" verdict becomes FLAG when one unsafe label is this close.
        self.flag_unsafe_peak = _env_float("JEV_FLAG_PEAK", 0.40)
        self.mode = (mode or os.getenv("JEV_STYLE_MODE") or "builtin").strip().lower()
        if self.mode not in {"builtin", "finetuned", "server", "auto"}:
            raise ValueError("JEV_STYLE_MODE must be builtin, finetuned, server or auto")
        self._server = server_client
        self._finetuned = finetuned
        # The fine-tuned model is weakest on harmful_instructions (59.75 % test
        # recall), so for that label Layer 1 / Layer 2 always backstop it.
        self.finetuned_min_confidence = _env_float("JEV_MODEL_MIN_CONFIDENCE", 0.55)
        # Carried risk: a FLAG becomes BLOCK while risk carried over from earlier
        # blocked turns of the chat is at or above this (see SessionMemory.carried_risk).
        self.carry_enabled = os.getenv("JEV_CARRY_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.carry_threshold = _env_float("JEV_CARRY_THRESHOLD", 0.50)

    def finetuned_model(self):
        if self._finetuned is None:
            from core.layers.jev_finetuned import FineTunedJEV

            self._finetuned = FineTunedJEV()
        return self._finetuned

    def describe(self) -> dict:
        info = {"mode": self.mode}
        if self.mode == "finetuned":
            info["finetuned"] = self.finetuned_model().check_config()
        return info

    # ------------------------------------------------------------------ public
    def evaluate(
        self,
        intent: IntentResult,
        guard: GuardVerdict | None = None,
        session_events: Sequence[Any] | None = None,
        prompt: str | None = None,
        carried_risk: float = 0.0,
    ) -> JEVResult:
        return self._apply_carried_risk(
            self._evaluate(intent, guard, session_events, prompt), intent, carried_risk)

    def _evaluate(self, intent, guard, session_events, prompt) -> JEVResult:
        if self.mode == "finetuned":
            if not prompt:
                raise JEVError("the fine-tuned JEV model needs the prompt text")
            try:
                label, confidence, probabilities = self.finetuned_model().classify(prompt)
            except Exception as error:
                # Never fall back to "safe": the pipeline blocks with jev_failed.
                raise JEVError(f"fine-tuned JEV unavailable: {error}") from error
            return self._combine_finetuned(intent, guard, session_events, label, confidence, probabilities)
        if self.mode in {"server", "auto"} and prompt:
            try:
                return self._evaluate_server(prompt, session_events, intent)
            except Exception as error:
                if self.mode == "server":
                    raise JEVError(f"Jev-Style server unavailable: {type(error).__name__}") from error
        return self._evaluate_builtin(intent, guard, session_events)

    # ------------------------------------------------------------ carried risk
    def _apply_carried_risk(self, result: JEVResult, intent: IntentResult, carried_risk: float) -> JEVResult:
        """Final verdict FLAG + carried risk >= threshold -> BLOCK.

        Applied after the model and the rules are combined, so a FLAG from either
        counts. Self-harm and distress turns are never upgraded.
        """
        try:
            carried = float(carried_risk or 0.0)
        except (TypeError, ValueError):
            carried = 0.0
        if (not self.carry_enabled or result.verdict != FLAG or carried < self.carry_threshold
                or "self_harm_risk" in {intent.intent, result.label}
                or "distress_signal" in result.rationale):
            return result
        result.verdict = BLOCK
        result.risk_score = round(min(1.0, max(result.risk_score, 0.7)), 4)
        result.risk_level = risk_from_score(result.risk_score)
        result.rationale = list(result.rationale) + ["escalated_carried_risk"]
        return result

    # --------------------------------------------------------------- finetuned
    def _combine_finetuned(self, intent, guard, session_events, label, confidence, probabilities):
        """Fine-tuned JEV classification + Layer 2 intent/confidence -> one verdict.

        The rule-based evaluation of the Layer 2 result is computed first and
        acts as a floor: the model can raise the verdict, never lower it. So a
        harmful_instructions prompt the model misses (its weakest class) is
        still flagged or blocked by Layer 1 / Layer 2.
        """
        started = time.perf_counter()
        base = self._evaluate_builtin(intent, guard, session_events)
        rank = {ALLOW: 0, FLAG: 1, BLOCK: 2}
        rationale = [f"jev_model={label}:{confidence:.2f}", f"layer2={intent.intent}:{intent.confidence:.2f}"]
        model_verdict = ALLOW
        if label in UNSAFE_INTENTS and confidence >= self.finetuned_min_confidence:
            agrees = label == intent.intent or label in {
                FINDING_LABEL[f] for f in (guard.findings if guard else ()) if f in FINDING_LABEL}
            if agrees or confidence >= self.block_confidence:
                model_verdict = BLOCK
                rationale.append("jev_model_unsafe" + ("_agrees_with_layers" if agrees else "_high_confidence"))
            else:
                model_verdict = FLAG
                rationale.append("jev_model_unsafe_disagrees_with_layer2")
        elif label == "safe" and base.verdict != ALLOW:
            rationale.append("layer_backstop_kept")
        verdict = max(base.verdict, model_verdict, key=rank.get)
        final_label = label if label in UNSAFE_INTENTS and model_verdict != ALLOW else base.label
        risk = max(base.risk_score, INTENT_BASE_RISK.get(label, 0.0) * confidence if label != "safe" else 0.0)
        return JEVResult(
            verdict=verdict,
            label=final_label,
            confidence=round(confidence if final_label == label else base.confidence, 4),
            risk_score=round(min(1.0, risk), 4),
            risk_level=risk_from_score(risk),
            rationale=rationale + base.rationale,
            model="guardgpt-jev-finetuned (Qwen3.5)",
            backend="finetuned",
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            probabilities=probabilities,
        )

    # ----------------------------------------------------------------- builtin
    @staticmethod
    def _recent_blocks(events: Sequence[Any] | None, window: int = 3) -> int:
        count = 0
        for event in list(events or ())[-window:]:
            if isinstance(event, Mapping):
                blocked = event.get("is_blocked", event.get("blocked", False))
                support = event.get("support", False)
            else:
                blocked = getattr(event, "is_blocked", False)
                support = getattr(event, "support", False)
            # A supportive self-harm reply is recorded as BLOCK but is not an attack.
            count += bool(blocked) and not support
        return count

    def _evaluate_builtin(
        self,
        intent: IntentResult,
        guard: GuardVerdict | None,
        session_events: Sequence[Any] | None,
    ) -> JEVResult:
        started = time.perf_counter()
        label = intent.intent
        confidence = float(intent.confidence)
        probabilities = intent.probabilities or {}
        unsafe_mass = sum(float(probabilities.get(name, 0.0)) for name in UNSAFE_INTENTS)
        max_unsafe = max((float(probabilities.get(name, 0.0)) for name in UNSAFE_INTENTS), default=0.0)
        # Expected risk under the Layer 2 distribution.
        risk = sum(float(probabilities.get(name, 0.0)) * INTENT_BASE_RISK[name] for name in UNSAFE_INTENTS)
        rationale: list[str] = []

        guard_labels = {FINDING_LABEL[f] for f in (guard.findings if guard else ()) if f in FINDING_LABEL}
        corroborated = label in guard_labels or (guard is not None and guard.status == UNSAFE)
        if guard is not None:
            risk = max(risk, guard.risk_score * 0.9)

        if label in UNSAFE_INTENTS:
            if not intent.passed_threshold:
                verdict = FLAG if unsafe_mass >= self.flag_unsafe_mass else ALLOW
                rationale.append("unsafe_intent_below_threshold")
            elif corroborated:
                verdict = BLOCK
                rationale.append("unsafe_intent_corroborated_by_layer1")
            elif confidence >= self.block_confidence:
                verdict = BLOCK
                rationale.append("unsafe_intent_high_confidence")
            else:
                verdict = FLAG
                rationale.append("unsafe_intent_uncorroborated")
        else:
            if guard is not None and guard.status == UNSAFE:
                # Layer 1 saw an explicit attack the classifier did not.
                verdict = BLOCK
                rationale.append("layer1_unsafe_overrides_safe_intent")
            elif max_unsafe >= self.flag_unsafe_peak:
                runner_up = max(UNSAFE_INTENTS, key=lambda name: float(probabilities.get(name, 0.0)))
                if runner_up == "self_harm_risk":
                    # Low mood rather than a request for harm: answer with care.
                    verdict = ALLOW
                    rationale.append("distress_signal")
                else:
                    verdict = FLAG
                    rationale.append("close_unsafe_runner_up")
            else:
                verdict = ALLOW
                rationale.append("intent_safe")

        # Session escalation: only ever stricter, and only for a turn that
        # itself carries unsafe signal. A clean "hello" after a blocked turn is
        # never blocked because of history.
        recent_blocks = self._recent_blocks(session_events)
        if recent_blocks >= 2 and verdict == FLAG:
            verdict = BLOCK
            rationale.append("escalated_repeated_unsafe_turns")
            risk = max(risk, 0.7)
        elif recent_blocks >= 2 and verdict == ALLOW and max_unsafe >= self.flag_unsafe_peak:
            # The strongest single unsafe label must lean unsafe. The summed mass is
            # not used: small scraps spread over several labels (the classifier's usual
            # noise on harmless harassment-awareness or security topics) must not flag.
            # A label at this bar is normally already FLAGged above (and upgraded to
            # BLOCK by the repeated-blocks rule), so this branch only adds the case the
            # runner-up rule deliberately lets through: a self-harm distress signal.
            verdict = FLAG
            rationale.append("escalated_after_blocked_turns")

        if label == "self_harm_risk" and verdict != ALLOW:
            rationale.append("support_route")
        elapsed = (time.perf_counter() - started) * 1000.0
        return JEVResult(
            verdict=verdict,
            label=label,
            confidence=round(confidence, 4),
            risk_score=round(min(1.0, risk), 4),
            risk_level=risk_from_score(risk),
            rationale=rationale,
            model=MODEL_NAME,
            backend="builtin",
            latency_ms=round(elapsed, 3),
        )

    # ------------------------------------------------------------------ server
    def _evaluate_server(self, prompt: str, session_events, intent: IntentResult) -> JEVResult:
        if self._server is None:
            from core.jev_style_client import JevStyleClient

            self._server = JevStyleClient()
        raw = self._server.decide(prompt, session_events)
        choice = str(raw.get("choice") or raw.get("decision") or "").strip()
        label = SERVER_LABELS.get(choice.lower())
        if label is None:
            raise JEVError(f"unexpected Jev-Style choice {choice!r}")
        confidence = float(raw.get("confidence") or 0.0)
        verdict = ALLOW if label == "safe" else BLOCK if confidence >= self.block_confidence else FLAG
        risk = 0.0 if label == "safe" else INTENT_BASE_RISK[label] * confidence
        return JEVResult(
            verdict=verdict,
            label=label,
            confidence=round(confidence, 4),
            risk_score=round(risk, 4),
            risk_level=risk_from_score(risk),
            rationale=["jev_style_server", f"layer2_intent={intent.intent}"],
            model=str(raw.get("model") or "jev-style"),
            backend="server",
            latency_ms=float(raw.get("latency_ms") or raw.get("elapsed_ms") or 0.0),
        )

"""Layer 3 - Guard / Policy Engine.

Combines the guard and intent signals (Layer 1 input verdict, Layer 2 intent,
JEV result) into one decision:

    ALLOW -> LangGraph agent -> Ollama / Llama 3.2 tool execution
    FLAG  -> review / escalate (queued for a human; no answer is released)
    BLOCK -> reject with a safe notice (or a supportive reply for self-harm)

The engine is pure: same inputs, same decision.
"""

from __future__ import annotations

import re

from core.layers.contracts import (
    ALLOW,
    BLOCK,
    FLAG,
    UNSAFE,
    GuardVerdict,
    IntentResult,
    JEVResult,
    PolicyDecision,
    risk_from_score,
)

BLOCK_MESSAGES = {
    "prompt_injection": "I can't follow instructions that try to override or reveal my configuration.",
    "jailbreak": "I can't switch off my safety guidelines, but I'm happy to help with something else.",
    "harmful_instructions": "I can't help with that because it could cause real harm to people.",
    "manipulation": "I can't help with manipulating or pressuring someone.",
    "invalid_input": "Please enter a prompt (maximum 6,000 UTF-8 bytes).",
}
DEFAULT_BLOCK_MESSAGE = "I can't help with this request because it was identified as unsafe."
_DEATH_OR_INJURY = re.compile(
    r"\b(?:die|dying|dead|death|suicid\w*|sucid\w*|sucide|kill(?:ing)? my ?self|kms|end (?:it|my life|everything)|"
    r"hurt(?:ing)? myself|harm(?:ing)? myself|cut(?:ting)? myself|overdose|not (?:be )?(?:here|alive)|"
    r"disappear|(?:not|never) wake up|better off without me|no reason to live|point in living|"
    r"(?:wasn't|weren't|was not|were not) (?:here|around|alive)|pills|jump(?:ing)? off|hang(?:ing)? myself)\b",
    re.IGNORECASE,
)

FLAG_MESSAGE = (
    "Your request has been flagged for review, so I'm not answering it right now. "
    "If you meant something harmless, try rephrasing it with a bit more context."
)


class PolicyEngine:
    def decide(
        self,
        guard: GuardVerdict,
        intent: IntentResult,
        jev: JEVResult,
        prompt_text: str | None = None,
    ) -> PolicyDecision:
        reasons: list[str] = []

        # Invalid input never reaches the other layers' verdicts.
        if {"empty_input", "input_too_long"} & set(guard.findings):
            return PolicyDecision(BLOCK, "safe", 0.0, [f for f in guard.findings], BLOCK_MESSAGES["invalid_input"],
                                  route="refusal")

        # Combine guard + intent signals into one risk score.
        risk = max(guard.risk_score, jev.risk_score)

        # Self-harm is a support need, not an attack: answer with care instead
        # of a refusal, unless the same message is also an attack. The crisis
        # support route is taken when the Layer 1 assessor found a crisis, when
        # JEV blocks a confident self_harm_risk label, or when JEV flags it and
        # the text refers to death or self-injury. Lower-confidence distress
        # ("my parents expect so much of me") keeps the label as a risk signal
        # but is answered normally in the agent's wellbeing mode.
        explicit = _DEATH_OR_INJURY.search(prompt_text or "") is not None
        self_harm = (
            guard.self_harm_level == "crisis"
            or (intent.intent == "self_harm_risk" and intent.passed_threshold
                and (jev.verdict == BLOCK or (jev.verdict == FLAG and explicit)))
        )
        attack = guard.status == UNSAFE and intent.intent != "self_harm_risk"
        if self_harm and not attack:
            reasons += ["self_harm_support"] + [f for f in guard.findings if f.startswith("self_harm")]
            return PolicyDecision(BLOCK, "critical" if guard.self_harm_level == "crisis" else "high",
                                  max(risk, 0.85), reasons, "", route="support")
        distress = (intent.intent == "self_harm_risk" and intent.passed_threshold) or (
            "distress_signal" in jev.rationale)
        if distress and not attack:
            # Distress without explicit risk: normal answer, wellbeing mode.
            reasons += ["emotional_distress", "wellbeing_mode"] + [f"jev:{r}" for r in jev.rationale]
            return PolicyDecision(ALLOW, "medium", max(risk, 0.4), reasons, "", route="agent")

        if guard.status == UNSAFE:
            reasons.append("layer1_unsafe")
            reasons += [f for f in guard.findings if f != "sensitive_data_redacted"]
        if jev.verdict == BLOCK:
            reasons.append("jev_block")
        elif jev.verdict == FLAG:
            reasons.append("jev_flag")
        reasons += [f"jev:{item}" for item in jev.rationale]

        if guard.status == UNSAFE or jev.verdict == BLOCK:
            action = BLOCK
        elif jev.verdict == FLAG:
            action = FLAG
        else:
            action = ALLOW
            # Passive ideation / distress / PII still get a normal, careful answer.
            if guard.status != "SAFE":
                reasons.append("layer1_flag_noted")

        label = intent.intent if intent.intent != "safe" else jev.label
        if action == BLOCK:
            message, route = BLOCK_MESSAGES.get(label, DEFAULT_BLOCK_MESSAGE), "refusal"
        elif action == FLAG:
            message, route = FLAG_MESSAGE, "review"
        else:
            message, route = "", "agent"
        return PolicyDecision(
            action=action,
            risk_level=risk_from_score(risk),
            risk_score=round(risk, 4),
            reason_codes=list(dict.fromkeys(reasons)),
            user_message=message,
            route=route,
        )

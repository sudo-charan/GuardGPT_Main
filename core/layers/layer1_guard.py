"""Layer 1 - Guard & Security.

Input mode (pre-generation)
    prompt analysis, prompt-injection detection, jailbreak detection, content
    moderation, attack / harm detection, sensitive-data check and an initial
    risk assessment.  Result: SAFE | FLAG | UNSAFE, passed on to Layer 2.

Output mode (post-generation)
    generated-content analysis, harmful-output detection, policy compliance
    (model reviewer), PII / data leakage, system-prompt leakage,
    response-intent consistency and a final risk assessment.
    Result: SAFE -> release | FLAG -> review | UNSAFE -> block / regenerate.

Layer 1 is deterministic apart from the optional model reviewer in output
mode. It never decides the final action on its own: its verdict is one of the
signals the Layer 3 policy engine combines.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from core import output_checks, self_harm
from core.jailbreak_patterns import _pattern_hits
from core.layers.contracts import FLAG, OUTPUT_ROUTE, SAFE, UNSAFE, GuardVerdict
from core.sensitive_data import redact

MAX_PROMPT_BYTES = 6000

# --------------------------------------------------------------------------
# Deterministic detectors. Each finding names the Layer 2 label it supports,
# so Intent Intelligence can use it as evidence instead of re-deriving it.
# --------------------------------------------------------------------------

FINDING_LABEL = {
    "instruction_override": "prompt_injection",
    "system_prompt_extraction": "prompt_injection",
    "embedded_instruction": "prompt_injection",
    "role_manipulation": "jailbreak",
    "safety_bypass": "jailbreak",
    "jailbreak_persona": "jailbreak",
    "credential_theft_request": "harmful_instructions",
    "auth_bypass_request": "harmful_instructions",
    "unauthorized_access_request": "harmful_instructions",
    "malware_request": "harmful_instructions",
    "weapon_or_violence_request": "harmful_instructions",
    "illegal_activity_request": "harmful_instructions",
    "coercion_request": "manipulation",
    "self_harm_crisis": "self_harm_risk",
    "self_harm_passive": "self_harm_risk",
}

# Findings that are attacks on the system itself: UNSAFE at input.
ATTACK_FINDINGS = frozenset({
    "instruction_override", "system_prompt_extraction", "embedded_instruction",
    "role_manipulation", "safety_bypass", "jailbreak_persona",
})
# Findings that are requests for harm: UNSAFE at input.
HARM_FINDINGS = frozenset({
    "credential_theft_request", "auth_bypass_request", "unauthorized_access_request",
    "malware_request", "weapon_or_violence_request", "illegal_activity_request",
    "coercion_request",
})

_JAILBREAK_PERSONA = re.compile(
    r"\b(?:do anything now|\bdan\b(?: mode)?|developer mode|jailbreak(?:ed)? mode|god mode|"
    r"aim,? an always intelligent|uncensored (?:ai|assistant|mode)|unrestricted (?:ai|assistant|mode)|"
    r"no (?:content )?(?:policy|policies|filters?|rules|restrictions|limits|guidelines)|"
    r"(?:free|freed) from (?:all )?(?:the )?(?:rules|guidelines|restrictions)|"
    r"(?:ignore|ignores|ignoring) (?:all )?(?:your |the )?(?:ethic(?:s|al)|moral|safety)[a-z ]{0,20}(?:guidelines|rules|policies)|"
    r"stay in character|never refuse|stop refusing|without (?:any )?(?:warnings|disclaimers|filters)|"
    r"(?:turn|switch) off (?:your )?(?:filters?|safety)|enable jailbreak|"
    r"bypass (?:your |the )?(?:review|filter|moderation|guard)[a-z ]{0,12}|"
    r"authori[sz]e you to (?:bypass|ignore|disable))\b"
)

_EMBEDDED_INSTRUCTION = re.compile(
    r"(?:\bprint|\brepeat|\boutput|\bshow)\s+(?:everything|all(?: the)? text|the text)\s+(?:above|before)|"
    r"\bnew instructions?\b.{0,40}\b(?:override|replace|ignore|void)|"
    r"\b(?:previous|prior|earlier) (?:instructions|rules|constraints|policy) (?:are|is) (?:void|cancel(?:l)?ed|replaced)|"
    r"\bfollow (?:any|the) instructions (?:inside|in) (?:it|this|the document)|"
    r"\bdisregard (?:prior|previous|all|your) (?:rules|guidelines|instructions)|"
    r"\bforget (?:everything|all) (?:you were told|above|before)|"
    r"\b(?:your|the) (?:initial|original|hidden|secret) (?:prompt|rules|instructions|key)\b"
)

_UNAUTHORIZED_ACCESS = re.compile(
    r"\b(?:hack|hacking|break into|get into|log ?in(?:to)?|access|take over|crack)\b[^.?!]{0,40}"
    r"\b(?:someone(?:'s| else's)|somebody(?:'s| elses?)|my (?:friend|ex|girlfriend|boyfriend|wife|husband|"
    r"classmate|neighbou?r|boss|coworker)s?'?s?|(?:his|her|their) )[^.?!]{0,30}"
    r"\b(?:account|instagram|insta|instahram|facebook|email|e-mail|gmail|phone|whatsapp|wifi|wi-fi|computer|password)s?\b"
    r"|\bhack (?:my |a |an |the )?(?:friend|ex|girlfriend|boyfriend|classmate|neighbou?r)s?'?s? \w+"
    r"|\bwithout (?:his|her|their) (?:knowledge|permission|knowing)\b[^.?!]{0,20}\b(?:account|phone|email)"
)

_MALWARE = re.compile(
    r"\b(?:write|create|build|make|code|develop|give me)\b[^.?!]{0,40}\b(?:ransomware|keylogger|malware|"
    r"trojan|spyware|botnet|virus that|rootkit|credential stealer|stealer|phishing (?:page|kit|site))\b"
    r"|\bddos (?:attack )?(?:against|on) \w+|\blaunch (?:a )?ddos\b"
    r"|\bsecretly\b[^.?!]{0,60}\b(?:cop(?:y|ies)|send(?:s)?|upload(?:s)?|exfiltrat\w*|steal(?:s)?|record(?:s)?)\b"
    r"|\bexfiltrat\w*\b"
)

_VIOLENCE = re.compile(
    r"\b(?:how (?:do|can|could|would|should) i|how to|best way to|help me|tell me how to|teach me (?:how )?to|"
    r"instructions (?:to|for)|step by step)\b[^.?!]{0,60}"
    r"\b(?:kill|murder|stab|shoot|poison|injure|hurt|maim|torture|kidnap|strangle|beat up|attack)\b"
    r"(?!\s+(?:myself|my ?self|me\b))[^.?!]{0,40}"
    r"|\b(?:make|build|assemble)\b[^.?!]{0,30}\b(?:bomb|explosive|pipe bomb|molotov|napalm|nerve agent|"
    r"chemical weapon|bioweapon|toxic gas|poison gas|ghost gun)\b"
    r"|\bwithout (?:leaving (?:any )?evidence|getting caught|being caught)\b"
    r"|\b(?:bleed out|untraceable poison|hard to detect in an autopsy)\b"
)

_ILLEGAL = re.compile(
    r"\b(?:make|cook|synthes[iz]e|produce)\b[^.?!]{0,30}\b(?:meth(?:amphetamine)?|cocaine|heroin|fentanyl|lsd)\b"
    r"|\b(?:steal|shoplift|hotwire)\b[^.?!]{0,30}\b(?:car|card|credit card|identity|money|store|shop)\b"
    r"|\bsteal a car\b|\bfake (?:id|passport)\b|\blaunder(?:ing)? money\b"
    r"|\b(?:share|post|leak)\b[^.?!]{0,30}\bprivate (?:photos?|pictures?|images?|videos?)\b"
    r"|\b(?:humiliating|degrading) sexual\b"
)

_COERCION = re.compile(
    r"\b(?:manipulat\w*|gaslight\w*|guilt[- ]?trip\w*|emotional(?:ly)? blackmail\w*|blackmail\w*|"
    r"love ?bomb\w*|groom\w*|coerc\w*|brainwash\w*)\b"
    r"|\b(?:pressure|trick|con|force|guilt)\b[^.?!]{0,30}\binto\b"
    r"|\bisolate (?:my |her |him |them )?\w+ from\b"
    r"|\bgain (?:their|his|her) trust without (?:their|his|her) parents\b"
    r"|\b(?:fake|false) (?:sob story|emergency|charity)\b"
)

_DEFENSIVE = re.compile(
    r"\b(?:protect|defend|prevent|secure|detect|recogni[sz]e|spot|avoid|report|warn|mitigat\w*|"
    r"stay safe|safeguard)\b"
)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    for quote in ("‘", "’", "`"):
        text = text.replace(quote, "'")
    for quote in ("“", "”", "«", "»"):
        text = text.replace(quote, '"')
    return " ".join(text.split())


def detect(prompt: str) -> list[str]:
    """Deterministic input findings for one prompt (no models)."""
    text = _normalise(prompt)
    findings = list(_pattern_hits(text))
    if _JAILBREAK_PERSONA.search(text):
        findings.append("jailbreak_persona")
    if _EMBEDDED_INSTRUCTION.search(text):
        findings.append("embedded_instruction")
    defensive = bool(_DEFENSIVE.search(text))
    if _UNAUTHORIZED_ACCESS.search(text) and not defensive:
        findings.append("unauthorized_access_request")
    if _MALWARE.search(text) and not defensive:
        findings.append("malware_request")
    if _VIOLENCE.search(text) and not defensive:
        findings.append("weapon_or_violence_request")
    if _ILLEGAL.search(text) and not defensive:
        findings.append("illegal_activity_request")
    if _COERCION.search(text) and not re.search(r"\b(?:am i being|is (?:he|she|my \w+) )\b", text):
        findings.append("coercion_request")
    return list(dict.fromkeys(findings))


class InputGuard:
    """Layer 1, input mode."""

    def __init__(self, dataset_provider=None) -> None:
        # ``dataset_provider`` returns a DatasetLoader-like object (query_top_k);
        # None disables content-moderation evidence (offline tests).
        self._dataset_provider = dataset_provider

    def check(
        self,
        prompt: str,
        self_harm_history: Iterable[str] = (),
        embedding: Any = None,
    ) -> GuardVerdict:
        checks: dict[str, Any] = {}
        if not isinstance(prompt, str) or not prompt.strip():
            return GuardVerdict("input", UNSAFE, 0.0, ["empty_input"], {"prompt_analysis": "empty"})
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            return GuardVerdict("input", UNSAFE, 0.0, ["input_too_long"], {"prompt_analysis": "too_long"})
        checks["prompt_analysis"] = {"chars": len(prompt), "lines": prompt.count("\n") + 1}

        # Sensitive data check: detected values never reach later layers.
        sensitive = redact(prompt)
        clean = sensitive.text if sensitive.detected else prompt
        checks["sensitive_data"] = {
            "detected": sensitive.detected, "pii_types": list(sensitive.pii_types),
            "span_count": sensitive.span_count,
        }

        findings = detect(clean)
        checks["prompt_injection"] = [f for f in findings if FINDING_LABEL.get(f) == "prompt_injection"]
        checks["jailbreak"] = [f for f in findings if FINDING_LABEL.get(f) == "jailbreak"]
        checks["attack_harm"] = [f for f in findings if f in HARM_FINDINGS]

        assessment = self_harm.assess(clean, self_harm_history)
        if assessment.is_crisis:
            findings.append("self_harm_crisis")
        elif assessment.level == "passive":
            findings.append("self_harm_passive")
        checks["self_harm"] = {"level": assessment.level, "reasons": list(assessment.reasons),
                               "follow_up": assessment.follow_up}

        evidence = self._content_moderation(clean, embedding)
        checks["content_moderation"] = evidence

        # Initial risk assessment.
        risk = 0.0
        if any(f in ATTACK_FINDINGS for f in findings):
            risk = max(risk, 0.9)
        if any(f in HARM_FINDINGS for f in findings):
            risk = max(risk, 0.85)
        if assessment.is_crisis:
            risk = max(risk, 0.9)
        elif assessment.at_risk:
            risk = max(risk, 0.6)
        elif assessment.level == "distress":
            risk = max(risk, 0.25)
        if sensitive.detected:
            risk = max(risk, 0.3)
        if evidence.get("label") not in (None, "safe") and evidence.get("similarity", 0.0) >= 0.85:
            # Near-duplicate of a known unsafe example.
            risk = max(risk, 0.7)
            findings.append("near_duplicate_unsafe_example")

        if any(f in ATTACK_FINDINGS or f in HARM_FINDINGS for f in findings):
            status = UNSAFE
        elif findings or sensitive.detected or assessment.level != "none":
            status = FLAG
        else:
            status = SAFE
        if sensitive.detected:
            findings.append("sensitive_data_redacted")
        return GuardVerdict(
            mode="input",
            status=status,
            risk_score=round(risk, 4),
            findings=list(dict.fromkeys(findings)),
            checks=checks,
            redacted_prompt=clean,
            pii_types=list(sensitive.pii_types),
            pii_span_count=sensitive.span_count,
            self_harm_level=assessment.level,
            dataset_evidence=evidence,
        )

    def _content_moderation(self, text: str, embedding: Any) -> dict:
        if self._dataset_provider is None:
            return {}
        try:
            loader = self._dataset_provider()
            if loader is None:
                return {}
            matches = loader.query_top_k(text, top_k=1, embedding=embedding)
        except Exception as error:  # evidence is optional, never fatal
            return {"error": type(error).__name__}
        if not matches:
            return {}
        best = matches[0]
        return {
            "label": best.get("intent"),
            "similarity": float(best.get("_similarity", 0.0)),
            "record_id": best.get("request_id"),
            "category_scores": best.get("category_scores", {}),
        }


# Model-reviewer categories that are hard safety failures.
HARD_REVIEW_CATEGORIES = frozenset({"harm", "jailbreak", "privacy", "hate", "sexual_abuse"})
_LEAK_FINDINGS = frozenset({"system_instruction_leak", "guardrail_disclosure_after_attack"})


class OutputGuard:
    """Layer 1, output mode."""

    def __init__(self, auditor=None) -> None:
        self.auditor = auditor  # OutputAuditor (Ollama reviewer) or None

    def check(
        self,
        prompt: str,
        candidate: str,
        *,
        intent: str = "safe",
        input_findings: Iterable[str] = (),
        request_span_digests: frozenset[str] = frozenset(),
        context: str | None = None,
    ) -> GuardVerdict:
        findings: list[str] = []
        checks: dict[str, Any] = {}
        attacks = [f for f in input_findings if f in ATTACK_FINDINGS]
        deterministic = output_checks.run(
            candidate,
            intent="self_harm" if context == "crisis_support" else intent,
            detected_attacks=attacks,
            request_span_digests=request_span_digests,
        )
        checks["deterministic"] = deterministic.as_dict()
        hard = soft = False

        # PII / data leakage.
        if deterministic.pii_leakage.status != output_checks.PASS:
            findings.append("pii_leakage")
            hard = True
        # System / prompt leakage and response-intent consistency.
        consistency = set(deterministic.intent_consistency.findings)
        if consistency & _LEAK_FINDINGS:
            findings.append("system_prompt_leakage")
            hard = True
        if "refusal_for_allowed_request" in consistency:
            findings.append("unnecessary_refusal")
            soft = True
        # Generated content analysis (format / degenerate output).
        if deterministic.format_structure.status != output_checks.PASS:
            findings.extend(deterministic.format_structure.findings)
            soft = True
        # Harmful output detection: the same deterministic detectors as input.
        output_harm = [f for f in detect(candidate) if f in HARM_FINDINGS and f != "coercion_request"]
        if output_harm and context != "crisis_support":
            findings.append("harmful_output_pattern")
            soft = True

        # Policy compliance: separate model reviewer (fails closed to review).
        if self.auditor is not None and not hard:
            try:
                verdict = self.auditor.review(prompt, candidate, context=context)
                checks["policy_compliance"] = {
                    "safe": verdict.safe, "relevant": verdict.relevant,
                    "categories": list(verdict.categories),
                }
                if not verdict.safe:
                    if set(verdict.categories) & HARD_REVIEW_CATEGORIES or not verdict.categories:
                        findings.append("policy_violation")
                        hard = True
                    else:
                        findings.append("policy_concern")
                        soft = True
                elif not verdict.relevant:
                    findings.append("relevance_unconfirmed")  # advisory only
            except Exception as error:
                checks["policy_compliance"] = {"error": type(error).__name__}
                findings.append("policy_review_unavailable")
                soft = True
        else:
            checks["policy_compliance"] = "skipped" if hard else "not_configured"

        status = UNSAFE if hard else FLAG if soft else SAFE
        risk = 0.9 if hard else 0.45 if soft else 0.05
        return GuardVerdict(
            mode="output",
            status=status,
            risk_score=risk,
            findings=list(dict.fromkeys(findings)),
            checks=checks,
            route=OUTPUT_ROUTE[status],
        )

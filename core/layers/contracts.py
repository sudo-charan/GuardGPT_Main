"""Typed hand-offs between the architecture layers.

Each layer produces exactly one of these objects. Downstream layers only read
them; nothing is re-classified further down the pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Layer 1 statuses (input and output mode).
SAFE = "SAFE"
FLAG = "FLAG"
UNSAFE = "UNSAFE"
GUARD_STATUSES = (SAFE, FLAG, UNSAFE)

# Output-mode routing for each Layer 1 status.
OUTPUT_ROUTE = {SAFE: "release", FLAG: "review", UNSAFE: "block_regenerate"}

# Layer 2 intent labels. These are exactly the six classes of the trained
# checkpoint and of the supplied dataset, and the six boxes of the diagram.
INTENT_LABELS = (
    "safe",
    "prompt_injection",
    "jailbreak",
    "harmful_instructions",
    "manipulation",
    "self_harm_risk",
)
UNSAFE_INTENTS = frozenset(INTENT_LABELS) - {"safe"}

# Layer 3 policy actions.
ALLOW = "ALLOW"
BLOCK = "BLOCK"
POLICY_ACTIONS = (ALLOW, FLAG, BLOCK)

RISK_LEVELS = ("safe", "low", "medium", "high", "critical")


def risk_from_score(score: float) -> str:
    """Map a 0..1 risk score onto the named risk levels."""
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.40:
        return "medium"
    if score >= 0.20:
        return "low"
    return "safe"


class _Serializable:
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardVerdict(_Serializable):
    """Layer 1 result, for either the input or the output mode."""

    mode: str                      # "input" | "output"
    status: str                    # SAFE | FLAG | UNSAFE
    risk_score: float = 0.0
    findings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    # Input mode only.
    redacted_prompt: str | None = None
    pii_types: list[str] = field(default_factory=list)
    pii_span_count: int = 0
    self_harm_level: str = "none"
    dataset_evidence: dict[str, Any] = field(default_factory=dict)
    # Output mode only.
    route: str | None = None       # release | review | block_regenerate

    @property
    def risk_level(self) -> str:
        return risk_from_score(self.risk_score)


@dataclass
class IntentResult(_Serializable):
    """Layer 2 result: intent label + confidence, with the evidence behind it."""

    intent: str
    confidence: float
    probabilities: dict[str, float]
    category: str                  # human-readable category name
    threshold: float               # acceptance bar the label had to clear
    passed_threshold: bool
    intent_risk: str               # intent risk mapping (safe..critical)
    sources: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class JEVResult(_Serializable):
    """JEV model output, consumed by the Layer 3 policy engine."""

    verdict: str                   # ALLOW | FLAG | BLOCK (recommendation)
    label: str                     # intent label the verdict is about
    confidence: float
    risk_score: float
    risk_level: str
    rationale: list[str] = field(default_factory=list)
    model: str = ""
    backend: str = "builtin"
    latency_ms: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)  # model's own distribution


@dataclass
class PolicyDecision(_Serializable):
    """Layer 3 Guard / Policy Engine decision."""

    action: str                    # ALLOW | FLAG | BLOCK
    risk_level: str
    risk_score: float
    reason_codes: list[str] = field(default_factory=list)
    user_message: str = ""
    route: str = "agent"           # agent | review | refusal | support

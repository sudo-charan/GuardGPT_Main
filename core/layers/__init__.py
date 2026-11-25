"""GuardGPT architecture layers.

The request path follows the architecture diagram exactly:

    Layer 1 Guard (input mode)  ->  Layer 2 Intent Intelligence  ->  JEV model
        ->  Layer 3 Guard / Policy Engine (ALLOW | FLAG | BLOCK)
        ->  LangGraph agent  ->  Ollama / Llama 3.2 tool execution
        ->  Layer 1 Guard (output mode: SAFE -> release | FLAG -> review |
            UNSAFE -> block / regenerate)

Every stage reads and writes the typed contracts in :mod:`core.layers.contracts`
and emits one audit-trail event carrying the request's trace id.
"""

from core.layers.contracts import (  # noqa: F401
    GUARD_STATUSES,
    INTENT_LABELS,
    POLICY_ACTIONS,
    GuardVerdict,
    IntentResult,
    JEVResult,
    PolicyDecision,
)

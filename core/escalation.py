"""Priority and escalation for FLAG review records.

A FLAG goes to the review queue. This decides how urgent it is and whether it is
escalated to a human straight away, from facts already known when it is queued:
the risk level, the intent, and how the session has gone.

    urgent    critical risk or a self-harm signal
    high      high risk, repeated flagged/blocked turns, or a risky session
    normal    everything else

``high`` and ``urgent`` are *escalated*: they are written to the escalation log,
counted in the metrics (an urgent one raises an alert) and listed first by
``python review_cli.py escalated``. Nothing here changes the FLAG decision and no
prompt text is stored.
"""

from __future__ import annotations

from typing import Iterable, Mapping

__all__ = ["PRIORITIES", "assess"]

PRIORITIES = ("normal", "high", "urgent")
REPEAT_WINDOW = 4          # previous turns looked at
REPEAT_THRESHOLD = 2       # flagged/blocked turns in that window
RISKY_SESSION = 0.60       # decayed session risk score


def assess(record: Mapping, recent_turns: Iterable[Mapping] = (), session_risk: float = 0.0) -> dict:
    """Return ``{"priority", "escalated", "escalation_reasons"}`` for one review record.

    ``recent_turns`` is the session's recent per-turn records, oldest first, with the
    current turn last (it is excluded from the repeat count).
    """
    reasons: list[str] = []
    priority = "normal"

    risk_level = str(record.get("risk_level") or "").lower()
    intent = str(record.get("intent") or "").lower()
    if risk_level == "critical":
        reasons.append("critical_risk")
        priority = "urgent"
    if intent == "self_harm_risk":
        reasons.append("self_harm_signal")
        priority = "urgent"

    if priority != "urgent":
        if risk_level == "high":
            reasons.append("high_risk")
            priority = "high"
        previous = list(recent_turns)[:-1][-REPEAT_WINDOW:]
        repeated = sum(1 for turn in previous if turn.get("action") in ("FLAG", "BLOCK"))
        if repeated >= REPEAT_THRESHOLD:
            reasons.append("repeated_flagged_turns")
            priority = "high"
        if session_risk >= RISKY_SESSION:
            reasons.append("risky_session")
            priority = "high"

    return {"priority": priority, "escalated": priority != "normal", "escalation_reasons": reasons}

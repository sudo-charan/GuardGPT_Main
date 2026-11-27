"""Bounded in-process conversation memory for one chat session.

It keeps compact per-turn *safety* records (intent, action, risk) that the JEV
model reads for escalation and the session report summarises. The answering
model additionally needs the conversation itself, otherwise a
follow-up such as "what did I say before?" or "summarise this chat" has nothing
to work with. This memory holds the last few released exchanges and the
self-harm risk level of each turn. It lives only in RAM, is capped in turns and
size, and is never written to the logs.

Only exchanges that were actually answered are remembered. A request that was
blocked is not replayed to the model on later turns, so an attack cannot be
smuggled forward through the history.
"""

from __future__ import annotations

import os
from collections import deque

MAX_TURNS = 8            # remembered user/assistant exchanges
MAX_MESSAGE_CHARS = 1500  # per stored message
MAX_LEVELS = 12           # remembered per-turn self-harm levels
MAX_TURN_RECORDS = 50     # remembered per-turn safety records
SESSION_RISK_DECAY = 0.6  # weight of older turns in the session risk score

# Carried risk (read by the JEV): a blocked turn raises it, every other turn fades it.
CARRY_DECAY = 0.9         # a blocked turn keeps at least this share of the old score
CARRY_SAFE_KEEP = 0.85    # share kept after a turn with risk 0 (a riskier turn keeps more)


def _env_fraction(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if 0.0 <= value <= 1.0 else default


class SessionMemory:
    def __init__(self) -> None:
        self._messages: deque[dict[str, str]] = deque(maxlen=MAX_TURNS * 2)
        self._levels: deque[str] = deque(maxlen=MAX_LEVELS)
        self._turns: deque[dict] = deque(maxlen=MAX_TURN_RECORDS)
        self.turn_count = 0
        # Sentence embedding of the previous prompt (RAM only), for follow-up detection.
        self.last_embedding = None
        self._carried_risk = 0.0
        self.carry_decay = _env_fraction("JEV_CARRY_DECAY", CARRY_DECAY)
        self.carry_safe_keep = _env_fraction("JEV_CARRY_SAFE_KEEP", CARRY_SAFE_KEEP)

    # ---------------------------------------------------------- safety records
    def record_turn(self, *, intent: str, confidence: float, action: str, risk_score: float,
                    topic: str | None = None, embedding=None, support: bool = False,
                    carried_escalation: bool = False) -> int:
        """Record one decided turn and return its 1-based index.

        ``support`` marks a supportive self-harm reply (recorded as BLOCK, but not an
        attack). ``carried_escalation`` marks a BLOCK that only the carried-risk rule
        caused; it fades the carried risk instead of raising it.
        """
        self.turn_count += 1
        if embedding is not None:
            self.last_embedding = embedding
        risk = min(1.0, max(0.0, float(risk_score)))
        self._turns.append({
            "turn_index": self.turn_count,
            "topic": str(topic) if topic else None,
            "intent": str(intent),
            "intent_confidence": round(float(confidence), 4),
            "action": str(action),
            "is_blocked": action == "BLOCK",
            "support": bool(support),
            "risk_score": round(risk, 4),
        })
        self._update_carried_risk(intent=str(intent), blocked=action == "BLOCK", risk=risk,
                                  skip=bool(support), carried_escalation=bool(carried_escalation))
        return self.turn_count

    def _update_carried_risk(self, *, intent: str, blocked: bool, risk: float, skip: bool,
                             carried_escalation: bool) -> None:
        """Blocked turn: max(risk, decay * score). Other turns: score * (keep + (1 - keep) * risk).

        Self-harm turns leave the score unchanged: a person who is struggling is not an attacker.
        """
        if skip or intent == "self_harm_risk":
            return
        score = self._carried_risk
        if blocked and not carried_escalation:
            score = max(risk, self.carry_decay * score)
        else:
            score = score * (self.carry_safe_keep + (1.0 - self.carry_safe_keep) * risk)
        self._carried_risk = min(1.0, max(0.0, score))

    def carried_risk(self) -> float:
        """Risk carried over from the earlier turns of this chat (0 until a turn is blocked)."""
        return round(self._carried_risk, 4)

    def recent_turns(self, limit: int = 5) -> list[dict]:
        return [dict(turn) for turn in list(self._turns)[-limit:]]

    def session_risk(self) -> float:
        """Exponentially decayed risk: recent turns count most, old ones fade.

        Unlike the old blocked-turn ratio this recovers: a session that had one
        blocked turn followed by ordinary chat drifts back towards zero.
        """
        score = 0.0
        for turn in self._turns:
            score = SESSION_RISK_DECAY * score + (1 - SESSION_RISK_DECAY) * turn["risk_score"]
        return round(score, 4)

    def report(self, session_id: str, request_id: str) -> dict:
        intents = [turn["intent"] for turn in self._turns]
        transitions = [
            {"from": previous, "to": current}
            for previous, current in zip(intents, intents[1:])
            if previous != current
        ]
        risk = self.session_risk()
        blocked = sum(turn["is_blocked"] for turn in self._turns)
        topics = [turn.get("topic") for turn in self._turns]
        return {
            "session_id": session_id,
            "request_id": request_id,
            "turn_count": self.turn_count,
            "topic_history": topics,
            "current_topic": topics[-1] if topics and topics[-1] else "unknown",
            "intent_history": intents,
            "current_intent": intents[-1] if intents else "unknown",
            "previous_intent": intents[-2] if len(intents) >= 2 else None,
            "transitions": transitions,
            "blocked_turns": blocked,
            "session_risk_score": risk,
            "carried_risk": self.carried_risk(),
            "session_summary": (
                f"{self.turn_count} turns, {blocked} blocked. Session risk {risk:.0%}. "
                f"Intent sequence: {', '.join(intents) or 'none'}."
            ),
        }

    @property
    def levels(self) -> list[str]:
        """Self-harm level of each previous turn, oldest first."""
        return list(self._levels)

    def record_level(self, level: str) -> None:
        self._levels.append(str(level))

    def messages(self) -> list[dict[str, str]]:
        """Previous exchanges as chat messages, oldest first."""
        return [dict(message) for message in self._messages]

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self._messages.append({"role": "user", "content": self._clip(user_text)})
        self._messages.append({"role": "assistant", "content": self._clip(assistant_text)})

    @staticmethod
    def _clip(text: str) -> str:
        text = str(text or "").strip()
        if len(text) <= MAX_MESSAGE_CHARS:
            return text
        return text[: MAX_MESSAGE_CHARS - 1].rstrip() + "\u2026"

    def __len__(self) -> int:
        return len(self._messages) // 2

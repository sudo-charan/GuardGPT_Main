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

from collections import deque

MAX_TURNS = 8            # remembered user/assistant exchanges
MAX_MESSAGE_CHARS = 1500  # per stored message
MAX_LEVELS = 12           # remembered per-turn self-harm levels
MAX_TURN_RECORDS = 50     # remembered per-turn safety records
SESSION_RISK_DECAY = 0.6  # weight of older turns in the session risk score


class SessionMemory:
    def __init__(self) -> None:
        self._messages: deque[dict[str, str]] = deque(maxlen=MAX_TURNS * 2)
        self._levels: deque[str] = deque(maxlen=MAX_LEVELS)
        self._turns: deque[dict] = deque(maxlen=MAX_TURN_RECORDS)
        self.turn_count = 0
        # Sentence embedding of the previous prompt (RAM only), for follow-up detection.
        self.last_embedding = None

    # ---------------------------------------------------------- safety records
    def record_turn(self, *, intent: str, confidence: float, action: str, risk_score: float,
                    topic: str | None = None, embedding=None) -> int:
        """Record one decided turn and return its 1-based index."""
        self.turn_count += 1
        if embedding is not None:
            self.last_embedding = embedding
        self._turns.append({
            "turn_index": self.turn_count,
            "topic": str(topic) if topic else None,
            "intent": str(intent),
            "intent_confidence": round(float(confidence), 4),
            "action": str(action),
            "is_blocked": action == "BLOCK",
            "risk_score": round(float(risk_score), 4),
        })
        return self.turn_count

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

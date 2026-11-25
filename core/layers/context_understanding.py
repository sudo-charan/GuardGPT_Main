"""Layer 2 context / task understanding.

Two things the safety label cannot say:

* **Task**: what the user wants done (question, how_to, creation, code,
  transformation, advice, chat, personal_support), matched against curated
  examples in ``intent_classifier/task_anchors.json`` with the sentence
  embedding Layer 2 already computed.
* **Conversation context**: is this a follow-up to the previous turn, did the
  topic change, and is the session getting riskier. It uses the bounded session
  memory only, so it adds no model call.

It is informational: it is logged under ``layer2_intent.context`` and shown in
verbose output, and it never changes an allow / flag / block decision. Any
failure returns ``None``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

TASKS_PATH = Path(__file__).resolve().parents[2] / "intent_classifier" / "task_anchors.json"
TOP_K = 2
TEMPERATURE = 0.05
MIN_SIMILARITY = 0.24
FOLLOW_UP_SIMILARITY = 0.50          # close to the previous prompt: same thread
FOLLOW_UP_WEAK_SIMILARITY = 0.25     # plus a reference word: a short follow-up
SHORT_MESSAGE_WORDS = 12
SESSION_WINDOW = 5

ELLIPTICAL_WORDS = 8                 # a message this short that refers back is a follow-up
_CONNECTOR = re.compile(r"^\s*(and|also|then|so|but|what about|how about|why not|ok(ay)?[, ]+so)\b", re.IGNORECASE)
_REFERENCE = re.compile(
    r"\b(it|its|that|this|those|these|them|they|he|she|again|more|also|previous|above|earlier|same|"
    r"another|next|else|then|continue|so far)\b", re.IGNORECASE)


class ContextAnalyzer:
    def __init__(self, encode: Callable[[list[str]], Any], path: Path = TASKS_PATH) -> None:
        self._encode = encode
        self.path = Path(path)
        self._lock = threading.Lock()
        self._vectors: dict[str, np.ndarray] | None = None

    def _load(self) -> dict[str, np.ndarray]:
        if self._vectors is not None:
            return self._vectors
        with self._lock:
            if self._vectors is None:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                vectors = {}
                for task, examples in data.items():
                    if task.startswith("_") or not isinstance(examples, list):
                        continue
                    texts = [str(item) for item in examples if str(item).strip()]
                    if texts:
                        vectors[task] = np.asarray(self._encode(texts), dtype=np.float32)
                self._vectors = vectors
        return self._vectors

    def _task(self, embedding: np.ndarray) -> tuple[str, float]:
        vectors = self._load()
        tasks = list(vectors)
        scores = np.array([float(np.sort(vectors[t] @ embedding)[::-1][:TOP_K].mean()) for t in tasks])
        logits = scores / TEMPERATURE
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        best = int(np.argmax(probabilities))
        task = tasks[best] if scores[best] >= MIN_SIMILARITY else "other"
        return task, float(probabilities[best])

    @staticmethod
    def session_signal(recent_turns: list[dict]) -> str:
        """clean / watch / escalating, from how many recent turns were not plain ALLOW-safe."""
        window = recent_turns[-SESSION_WINDOW:]
        risky = sum(1 for turn in window
                    if turn.get("action") != "ALLOW" or turn.get("intent") not in (None, "safe"))
        return "clean" if risky == 0 else "watch" if risky == 1 else "escalating"

    def analyze(self, prompt: str, embedding: np.ndarray | None, topic: str | None,
                memory: Any = None) -> dict | None:
        """Return the context block, or None when it cannot be computed."""
        if embedding is None:
            return None
        try:
            task, confidence = self._task(embedding)
            recent = memory.recent_turns(SESSION_WINDOW) if memory is not None else []
            previous = recent[-1] if recent else None
            previous_embedding = getattr(memory, "last_embedding", None) if memory is not None else None
            similarity = None
            if previous_embedding is not None:
                similarity = float(np.dot(previous_embedding, embedding))
            words = len(prompt.split())
            has_reference = bool(_REFERENCE.search(prompt)) and words <= SHORT_MESSAGE_WORDS
            # Elliptical follow-ups ("and what about reversing it?") share almost no
            # vocabulary with the earlier turn, so the embedding alone cannot see them.
            elliptical = words <= ELLIPTICAL_WORDS and (
                bool(_CONNECTOR.search(prompt)) or bool(_REFERENCE.search(prompt)))
            follow_up = bool(previous) and (
                elliptical
                or (similarity is not None and (
                    similarity >= FOLLOW_UP_SIMILARITY
                    or (has_reference and similarity >= FOLLOW_UP_WEAK_SIMILARITY))))
            previous_topic = (previous or {}).get("topic")
            topic_shift = bool(previous) and not follow_up and bool(previous_topic) and bool(topic) \
                and previous_topic != topic
            signal = self.session_signal(recent)
            parts = [f"{task.replace('_', ' ')}", "follow-up" if follow_up else None,
                     f"topic changed from {previous_topic}" if topic_shift else None,
                     f"session {signal}" if signal != "clean" else None]
            return {
                "task": task,
                "task_confidence": round(confidence, 4),
                "follow_up": follow_up,
                "topic_shift": topic_shift,
                "previous_topic": previous_topic,
                "similarity_to_previous": None if similarity is None else round(similarity, 4),
                "turns_so_far": int(getattr(memory, "turn_count", 0)) if memory is not None else 0,
                "session_signal": signal,
                "summary": ", ".join(part for part in parts if part),
            }
        except Exception as error:  # informational only: never break a request
            logger.warning("Context understanding unavailable: %s", error)
            return None

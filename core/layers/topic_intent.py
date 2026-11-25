"""Topic intent: what the user is talking about (coding, education, greeting ...).

This is deliberately separate from Layer 2's *safety* intent (safe, jailbreak,
...). It reuses the sentence embedding Layer 2 already computed, compares it with
curated examples in ``intent_classifier/topic_anchors.json`` and returns the
closest topic. It is informational only: it is logged and shown, and it never
changes an allow / flag / block decision. Any failure returns ``None``.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

TOPICS_PATH = Path(__file__).resolve().parents[2] / "intent_classifier" / "topic_anchors.json"
TOP_K = 2               # mean of the best K similarities per topic
TEMPERATURE = 0.05      # softmax sharpness over topic similarities
MIN_SIMILARITY = 0.24   # below this the message matches no topic well
GENERAL = "general"


class TopicIntent:
    def __init__(self, encode: Callable[[list[str]], Any], path: Path = TOPICS_PATH) -> None:
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
                for topic, examples in data.items():
                    if topic.startswith("_") or not isinstance(examples, list):
                        continue
                    texts = [str(item) for item in examples if str(item).strip()]
                    if texts:
                        vectors[topic] = np.asarray(self._encode(texts), dtype=np.float32)
                self._vectors = vectors
        return self._vectors

    def classify(self, embedding: np.ndarray | None) -> dict | None:
        """Return ``{"topic", "confidence", "similarity", "alternatives"}`` or None."""
        if embedding is None:
            return None
        try:
            vectors = self._load()
            topics = list(vectors)
            scores = np.array([
                float(np.sort(vectors[t] @ embedding)[::-1][:TOP_K].mean()) for t in topics
            ])
        except Exception as error:  # informational only: never break a request
            logger.warning("Topic intent unavailable: %s", error)
            return None
        logits = scores / TEMPERATURE
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities)
        best = int(order[0])
        topic = topics[best] if scores[best] >= MIN_SIMILARITY else GENERAL
        return {
            "topic": topic,
            "confidence": round(float(probabilities[best]), 4),
            "similarity": round(float(scores[best]), 4),
            "alternatives": [{"topic": topics[i], "confidence": round(float(probabilities[i]), 4)}
                             for i in order[1:3]],
        }

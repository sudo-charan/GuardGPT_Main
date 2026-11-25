"""Layer 2 - Intent Intelligence.

Produces one of the six architecture labels

    safe | prompt_injection | jailbreak | harmful_instructions | manipulation | self_harm_risk

plus a confidence and the full probability distribution, and hands
``intent + confidence`` to the JEV model.

Why an ensemble
---------------
The supplied checkpoint (`intent_classifier/best_model.pt`) is a fine-tuned
MiniLM with a 6-way head, trained on `data/guardgpt_dataset.jsonl`. That
dataset is noisy: its `manipulation` class is mostly profanity and slang, its
`jailbreak` class is mostly red-team security questions, and greetings or short
chat never appear as `safe`. On its own the head therefore scores "hi" as
manipulation (0.61), every jailbreak as prompt_injection, and "how do I protect
my account from phishing" as harmful_instructions (0.92). The old code hid this
behind keyword overrides and a separate 12-label prototype classifier, and the
JEV step collapsed anything that was not an attack into "Safe/Educational".

This layer combines four evidence sources into one distribution:

1. ``head``    the trained checkpoint's softmax (fine-tuned encoder),
2. ``anchors`` similarity to curated reference prompts per label
               (`intent_classifier/intent_anchors.json`, editable, no retraining),
3. ``dataset`` similarity-weighted vote of the nearest FAISS neighbours
               in the supplied 19,200-record dataset,
4. ``guard``   Layer 1's deterministic findings (pattern hits, self-harm
               assessor), which name the label they support.

Every source is optional, so the layer degrades gracefully (for example when
the embedding model is not cached) and records which sources contributed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from core.layers.contracts import INTENT_LABELS, UNSAFE_INTENTS, IntentResult, risk_from_score
from core.layers.layer1_guard import FINDING_LABEL

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "intent_classifier"
ANCHORS_PATH = MODEL_DIR / "intent_anchors.json"

CATEGORY_NAMES = {
    "safe": "Safe / benign request",
    "prompt_injection": "Prompt injection",
    "jailbreak": "Jailbreak attempt",
    "harmful_instructions": "Harmful instructions",
    "manipulation": "Manipulation / coercion",
    "self_harm_risk": "Self-harm risk",
}

# Ensemble weights. Tuned on a probe set and checked on a separate held-out
# set (see tests/test_intent_accuracy.py); override with GUARDGPT_L2_WEIGHTS,
# e.g. "head=0.2,anchors=0.6,dataset=0.2".
DEFAULT_WEIGHTS = {"head": 0.20, "anchors": 0.60, "dataset": 0.20}
# Probability mass each Layer 1 finding adds to its label before renormalising.
GUARD_EVIDENCE_WEIGHT = 0.60

# Threshold evaluation: the top label must clear its bar, otherwise the result
# is reported as below threshold and JEV treats it as uncertain.
LABEL_THRESHOLDS = {
    "safe": 0.40,
    "prompt_injection": 0.45,
    "jailbreak": 0.45,
    "harmful_instructions": 0.45,
    "manipulation": 0.45,
    "self_harm_risk": 0.40,
}

# Intent risk mapping (used for the result and by JEV).
INTENT_BASE_RISK = {
    "safe": 0.0,
    "prompt_injection": 0.85,
    "jailbreak": 0.85,
    "harmful_instructions": 0.90,
    "manipulation": 0.75,
    "self_harm_risk": 0.90,
}

ANCHOR_TOP_K = 3
ANCHOR_TEMPERATURE = 0.04
DATASET_TOP_K = 15
DATASET_MIN_SIMILARITY = 0.35


def _weights() -> dict[str, float]:
    raw = os.getenv("GUARDGPT_L2_WEIGHTS")
    if not raw:
        return dict(DEFAULT_WEIGHTS)
    weights = dict(DEFAULT_WEIGHTS)
    for part in raw.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            try:
                weights[key.strip()] = float(value)
            except ValueError:
                pass
    return weights


def _as_distribution(values: Iterable[float]) -> np.ndarray:
    array = np.clip(np.asarray(list(values), dtype=np.float64), 0.0, None)
    total = array.sum()
    return array / total if total > 0 else np.full(len(INTENT_LABELS), 1.0 / len(INTENT_LABELS))


# --------------------------------------------------------------------------
# Source 1: the trained checkpoint
# --------------------------------------------------------------------------

class TrainedIntentHead:
    """Fine-tuned MiniLM encoder + 6-way linear head from best_model.pt.

    The encoder is rebuilt from `intent_classifier/config.json` and the
    checkpoint's own weights, so it does not need the base model download and
    never shares (or perturbs) the encoder used for the FAISS index.
    """

    def __init__(self, model_dir: Path = MODEL_DIR, checkpoint: Path | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.checkpoint = Path(
            checkpoint or os.getenv("GUARDGPT_INTENT_MODEL") or self.model_dir / "best_model.pt"
        )
        self._lock = threading.Lock()
        self._loaded = False
        self._encoder = None
        self._head = None
        self._tokenizer = None
        self._labels: list[str] = []
        self.status: dict[str, Any] = {"active": False, "reason": "not_loaded"}

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                import torch
                from transformers import AutoTokenizer, BertConfig, BertModel

                checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
                labels = [str(label) for label in checkpoint["labels"]]
                if sorted(labels) != sorted(INTENT_LABELS):
                    raise ValueError(f"checkpoint labels {labels} do not match the architecture labels")
                state = checkpoint["model_state_dict"]
                config = BertConfig.from_json_file(str(self.model_dir / "config.json"))
                encoder = BertModel(config, add_pooling_layer=False)
                encoder_state = {
                    key.removeprefix("encoder."): value
                    for key, value in state.items()
                    if key.startswith("encoder.") and not key.startswith("encoder.pooler.")
                }
                missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
                missing = [key for key in missing if "position_ids" not in key]
                if missing:
                    raise ValueError(f"checkpoint is missing encoder weights: {missing[:3]}")
                weight, bias = state["classifier.weight"], state["classifier.bias"]
                head = torch.nn.Linear(weight.shape[1], weight.shape[0])
                head.load_state_dict({"weight": weight, "bias": bias})
                encoder.eval()
                head.eval()
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self._encoder, self._head, self._labels = encoder, head, labels
                self.status = {"active": True, "reason": "ok", "labels": labels,
                               "path": str(self.checkpoint)}
            except Exception as error:
                logger.warning("Trained intent head unavailable: %s", error)
                self.status = {"active": False, "reason": f"{type(error).__name__}: {error}"[:200],
                               "path": str(self.checkpoint)}

    @property
    def active(self) -> bool:
        self._load()
        return self._head is not None

    def probabilities(self, text: str) -> np.ndarray | None:
        """Distribution over INTENT_LABELS (architecture order) or None."""
        self._load()
        if self._head is None:
            return None
        import torch

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        chunks = [token_ids[start:start + 254] for start in range(0, max(len(token_ids), 1), 254)] or [[]]
        rows = []
        for chunk in chunks:
            encoded = self._tokenizer(
                self._tokenizer.decode(chunk), return_tensors="pt", truncation=True, max_length=256
            )
            with torch.no_grad():
                hidden = self._encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                rows.append(torch.softmax(self._head(pooled), dim=-1)[0].numpy())
        probs = np.asarray(rows)
        # A long prompt is as unsafe as its most unsafe chunk.
        combined = probs.mean(axis=0)
        for index, label in enumerate(self._labels):
            if label != "safe":
                combined[index] = probs[:, index].max()
        by_label = dict(zip(self._labels, combined))
        return _as_distribution(by_label[label] for label in INTENT_LABELS)


# --------------------------------------------------------------------------
# Source 2: curated anchors   Source 3: dataset neighbours
# --------------------------------------------------------------------------

class AnchorIndex:
    def __init__(self, encode: Callable[[list[str]], np.ndarray], path: Path = ANCHORS_PATH) -> None:
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
                for label in INTENT_LABELS:
                    examples = [str(item) for item in data.get(label, []) if str(item).strip()]
                    if examples:
                        vectors[label] = np.asarray(self._encode(examples), dtype=np.float32)
                self._vectors = vectors
        return self._vectors

    def probabilities(self, embedding: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        vectors = self._load()
        scores = {}
        for label in INTENT_LABELS:
            matrix = vectors.get(label)
            if matrix is None or not len(matrix):
                scores[label] = -1.0
                continue
            sims = np.sort(matrix @ embedding)[::-1][:ANCHOR_TOP_K]
            scores[label] = float(sims.mean())
        logits = np.array([scores[label] for label in INTENT_LABELS]) / ANCHOR_TEMPERATURE
        logits -= logits.max()
        return _as_distribution(np.exp(logits)), {k: round(v, 4) for k, v in scores.items()}


def dataset_probabilities(loader: Any, text: str, embedding: np.ndarray) -> tuple[np.ndarray | None, dict]:
    matches = loader.query_top_k(text, top_k=DATASET_TOP_K, embedding=embedding)
    votes = np.zeros(len(INTENT_LABELS))
    best = None
    for match in matches:
        similarity = float(match.get("_similarity", 0.0))
        label = match.get("intent")
        if label not in INTENT_LABELS or similarity < DATASET_MIN_SIMILARITY:
            continue
        best = best or match
        votes[INTENT_LABELS.index(label)] += similarity ** 4
    if not votes.any():
        return None, {"top_similarity": float(matches[0]["_similarity"]) if matches else 0.0}
    info = {
        "top_similarity": float(best["_similarity"]),
        "top_label": best.get("intent"),
        "top_record_id": best.get("request_id"),
        "neighbours": len(matches),
    }
    return _as_distribution(votes), info


# --------------------------------------------------------------------------
# Layer 2
# --------------------------------------------------------------------------

class IntentIntelligence:
    def __init__(
        self,
        head: TrainedIntentHead | None = None,
        encoder_provider: Callable[[], Any] | None = None,
        dataset_provider: Callable[[], Any] | None = None,
        anchors_path: Path = ANCHORS_PATH,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.head = head if head is not None else TrainedIntentHead()
        self._encoder_provider = encoder_provider
        self._dataset_provider = dataset_provider
        self._anchors = AnchorIndex(self._encode, anchors_path)
        self.weights = weights or _weights()
        self._last_embedding: np.ndarray | None = None

    # -- embedding (base MiniLM, the same space as the FAISS index)
    def _encoder(self):
        return self._encoder_provider() if self._encoder_provider else None

    def _encode(self, texts: list[str]) -> np.ndarray:
        encoder = self._encoder()
        if encoder is None:
            raise RuntimeError("embedding model unavailable")
        return np.asarray(
            encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32
        )

    def embed(self, text: str) -> np.ndarray | None:
        try:
            vector = self._encode([text.strip()])[0]
        except Exception as error:
            logger.warning("Embedding unavailable for Layer 2: %s", error)
            return None
        self._last_embedding = vector
        return vector

    @property
    def last_embedding(self) -> np.ndarray | None:
        return self._last_embedding

    def status(self) -> dict:
        return {"head": self.head.status if self.head is not None else {"active": False},
                "weights": self.weights}

    # -- main entry point
    def analyze(
        self,
        prompt: str,
        guard_findings: Iterable[str] = (),
        embedding: np.ndarray | None = None,
    ) -> IntentResult:
        text = (prompt or "").strip()
        if not text:
            probabilities = {label: 0.0 for label in INTENT_LABELS}
            probabilities["safe"] = 1.0
            return IntentResult("safe", 0.0, probabilities, CATEGORY_NAMES["safe"],
                                LABEL_THRESHOLDS["safe"], False, "safe", {}, ["empty_input"])

        sources: dict[str, Any] = {}
        parts: list[tuple[float, np.ndarray]] = []

        head_probs = self.head.probabilities(text) if self.head is not None else None
        if head_probs is not None:
            parts.append((self.weights.get("head", 0.0), head_probs))
            sources["head"] = self._rounded(head_probs)

        if embedding is None:
            embedding = self.embed(text)
        else:
            self._last_embedding = embedding
        if embedding is not None:
            try:
                anchor_probs, anchor_scores = self._anchors.probabilities(embedding)
                parts.append((self.weights.get("anchors", 0.0), anchor_probs))
                sources["anchors"] = self._rounded(anchor_probs)
                sources["anchor_similarity"] = anchor_scores
            except Exception as error:
                sources["anchors_error"] = type(error).__name__
            loader = self._dataset_provider() if self._dataset_provider else None
            if loader is not None:
                try:
                    dataset_probs, info = dataset_probabilities(loader, text, embedding)
                    sources["dataset_neighbours"] = info
                    if dataset_probs is not None:
                        parts.append((self.weights.get("dataset", 0.0), dataset_probs))
                        sources["dataset"] = self._rounded(dataset_probs)
                except Exception as error:
                    sources["dataset_error"] = type(error).__name__

        total_weight = sum(weight for weight, _ in parts)
        if total_weight > 0:
            combined = sum(weight * probs for weight, probs in parts) / total_weight
        else:
            combined = np.full(len(INTENT_LABELS), 1.0 / len(INTENT_LABELS))

        # Layer 1 evidence: each deterministic finding supports one label.
        reasons: list[str] = []
        supported = {FINDING_LABEL[f] for f in guard_findings if f in FINDING_LABEL}
        if supported:
            boost = np.zeros(len(INTENT_LABELS))
            for label in supported:
                boost[INTENT_LABELS.index(label)] = GUARD_EVIDENCE_WEIGHT / len(supported)
            combined = combined * (1.0 - GUARD_EVIDENCE_WEIGHT) + boost
            sources["guard"] = sorted(supported)
            reasons.append("layer1_evidence")
        combined = _as_distribution(combined)

        index = int(np.argmax(combined))
        label = INTENT_LABELS[index]
        confidence = float(combined[index])
        threshold = LABEL_THRESHOLDS[label]
        passed = confidence >= threshold
        if not parts:
            reasons.append("no_model_evidence")
        if not passed:
            reasons.append("below_threshold")
        risk = INTENT_BASE_RISK[label] * (confidence if label in UNSAFE_INTENTS else 1.0)
        return IntentResult(
            intent=label,
            confidence=round(confidence, 4),
            probabilities=self._rounded(combined),
            category=CATEGORY_NAMES[label],
            threshold=threshold,
            passed_threshold=passed,
            intent_risk=risk_from_score(risk),
            sources=sources,
            reasons=reasons,
        )

    @staticmethod
    def _rounded(probs: np.ndarray) -> dict[str, float]:
        return {label: round(float(value), 4) for label, value in zip(INTENT_LABELS, probs)}


def validate_checkpoint(path: str | Path | None = None) -> dict:
    """Cheap structural check of the trained head for `--status` and `health`."""
    target = Path(path or os.getenv("GUARDGPT_INTENT_MODEL") or MODEL_DIR / "best_model.pt")
    result = {"path": str(target), "ok": False, "reason": "missing", "labels": []}
    if not target.is_file():
        return result
    try:
        import torch

        checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as error:
        result["reason"] = f"unreadable: {type(error).__name__}"
        return result
    labels = checkpoint.get("labels") if isinstance(checkpoint, dict) else None
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(labels, list) or sorted(map(str, labels)) != sorted(INTENT_LABELS):
        result["reason"] = "labels_do_not_match_architecture"
        return result
    if not isinstance(state, dict) or "classifier.weight" not in state:
        result["reason"] = "incomplete_head"
        return result
    if tuple(state["classifier.weight"].shape) != (len(labels), 384):
        result["reason"] = "head_shape_mismatch"
        return result
    if not (MODEL_DIR / "config.json").is_file() or not (MODEL_DIR / "tokenizer.json").is_file():
        result["reason"] = "encoder_config_or_tokenizer_missing"
        return result
    result.update(ok=True, reason="ok", labels=[str(label) for label in labels])
    return result

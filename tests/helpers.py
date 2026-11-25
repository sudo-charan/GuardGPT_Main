"""Offline doubles for pipeline tests (no models, no Ollama)."""

from __future__ import annotations

import json
from pathlib import Path

from agent.graph import GuardAgent
from core.complete_pipeline import AuditLog, CompletePipeline
from core.layers.contracts import INTENT_LABELS, IntentResult
from core.layers.jev import JEVModel
from core.layers.layer1_guard import InputGuard, OutputGuard
from core.layers.layer2_intent import CATEGORY_NAMES, LABEL_THRESHOLDS
from core.layers.layer3_policy import PolicyEngine
from core.output_auditor import AuditVerdict
from core.review import ReviewQueue


class StubIntentLayer:
    """Layer 2 double: returns a fixed (or keyword-chosen) label."""

    def __init__(self, label="safe", confidence=0.9, by_keyword=None):
        self.label, self.confidence = label, confidence
        self.by_keyword = by_keyword or {}
        self.calls = []

    def embed(self, text):
        return None

    def analyze(self, prompt, guard_findings=(), embedding=None):
        self.calls.append((prompt, tuple(guard_findings)))
        label, confidence = self.label, self.confidence
        for keyword, (kw_label, kw_conf) in self.by_keyword.items():
            if keyword in prompt.lower():
                label, confidence = kw_label, kw_conf
        rest = (1 - confidence) / (len(INTENT_LABELS) - 1)
        probabilities = {name: (confidence if name == label else rest) for name in INTENT_LABELS}
        return IntentResult(label, confidence, probabilities, CATEGORY_NAMES[label],
                            LABEL_THRESHOLDS[label], confidence >= LABEL_THRESHOLDS[label], "safe",
                            {"stub": True}, [])


class ScriptedBackend:
    """Ollama double. ``answers`` are returned in order for generation calls."""

    model = "fixture"

    def __init__(self, answers=("A helpful answer.",)):
        self.answers = list(answers)
        self.calls = []

    def generate(self, prompt, system_prompt=None, schema=None, temperature=None, history=None):
        self.calls.append({"prompt": prompt, "system": system_prompt, "history": history, "schema": schema})
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]


class FixedAuditor:
    def __init__(self, verdict=None):
        self.verdict = verdict or AuditVerdict(safe=True, relevant=True, categories=[])
        self.calls = 0

    def review(self, prompt, candidate, context=None):
        self.calls += 1
        return self.verdict


def make_pipeline(root, intent=None, backend=None, auditor=None, jev=None):
    root = Path(root)
    backend = backend or ScriptedBackend()
    auditor = auditor or FixedAuditor()
    return CompletePipeline(
        input_guard=InputGuard(),
        intent_layer=intent or StubIntentLayer(),
        jev=jev or JEVModel(mode="builtin"),
        policy=PolicyEngine(),
        agent=GuardAgent(backend, OutputGuard(auditor)),
        audit_log=AuditLog(root / "guardgpt_audit.jsonl"),
        review_queue=ReviewQueue(root / "guardgpt_review.jsonl"),
    )


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

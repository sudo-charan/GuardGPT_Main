"""JEV adapter for the fine-tuned GuardGPT checkpoint (JEV_ADAPTER).

Uses the official Jev-Style runtime (``jev_style_decision.py`` from the base model
folder) for rendering, tokenisation and the yes/no verdict readout, loads the
fine-tuned weights from ``best_model.pt`` into it, and asks the same question
with the same class descriptions that were stored in the checkpoint at training
time. Nothing here is guessed: the question and options come from the checkpoint.

    JEV_STYLE_MODE=finetuned
    JEV_CHECKPOINT=.../intent_classifier/jev_finetuned/best_model.pt
    JEV_BASE_MODEL_DIR=<folder with model.safetensors-config, tokenizer.json,
                        readout_config.json and jev_style_decision.py>
    JEV_ADAPTER=.../jev_adapter_guardgpt.py

``JEV_STATE_FORMAT`` selects how the prompt is passed as the "state":
``text`` (the raw prompt string, default) or ``json`` ({"prompt": ...}).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

LABELS = ("safe", "prompt_injection", "jailbreak", "harmful_instructions", "manipulation", "self_harm_risk")


def _load_runtime(base_dir: Path):
    path = base_dir / "jev_style_decision.py"
    if not path.is_file():
        raise FileNotFoundError(f"jev_style_decision.py not found in JEV_BASE_MODEL_DIR: {base_dir}")
    spec = importlib.util.spec_from_file_location("jev_style_decision", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["jev_style_decision"] = module   # dataclass/pickle-safe
    spec.loader.exec_module(module)
    return module


class Scorer:
    def __init__(self, engine, question: dict, state_format: str) -> None:
        self.engine, self.question, self.state_format = engine, question, state_format

    def scores(self, text: str) -> dict[str, float]:
        state = {"prompt": text} if self.state_format == "json" else text
        # temperature=1.0 -> raw (yes - no) scores; GuardGPT applies the softmax.
        result = self.engine.decide(state, self.question, temperature=1.0)
        return {label: float(result["scores"][label]) for label in LABELS}


def load(checkpoint_path, base_model_dir, device="cpu"):
    import torch

    base = Path(base_model_dir)
    runtime = _load_runtime(base)
    engine = runtime.JevStyleDecision(base, device=device, dtype="float32")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    result = engine.model.load_state_dict(state, strict=False)
    own = set(engine.model.state_dict())
    unexpected = [key for key in result.unexpected_keys]
    # Tied embeddings: lm_head may legitimately be absent from `missing`.
    missing = [key for key in result.missing_keys if key not in {"lm_head.weight"}]
    if unexpected or missing or not own:
        raise RuntimeError(f"checkpoint does not match the base architecture "
                           f"(missing={missing[:3]}, unexpected={unexpected[:3]})")
    engine.model.to(engine.device).eval()
    # The readout direction depends on the (fine-tuned) output embeddings.
    weights = engine.model.get_output_embeddings().weight
    engine.direction = (weights[engine.renderer.yes].float() - weights[engine.renderer.no].float()).detach()

    classes = list(checkpoint.get("class_names") or LABELS)
    if tuple(classes) != LABELS:
        raise RuntimeError(f"checkpoint classes {classes} differ from GuardGPT labels {list(LABELS)}")
    question = {"t": "choice", "ins": checkpoint["question"],
                "crit": {name: checkpoint["options"][name] for name in LABELS}}
    return Scorer(engine, question, os.getenv("JEV_STATE_FORMAT", "text").strip().lower())

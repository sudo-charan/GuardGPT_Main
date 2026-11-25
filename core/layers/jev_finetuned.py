"""Fine-tuned JEV model backend (JEV_STYLE_MODE=finetuned).

Loads the team's fully fine-tuned JEV checkpoint (Qwen3.5 base, 24 layers,
~1.5 GB ``best_model.pt``) from a LOCAL path and scores the six GuardGPT
classes with the JEV yes/no verdict mechanism:

    class score = hidden state at the class's verdict slot
                  . (embedding[yes] - embedding[no])

then argmax over the six classes. The checkpoint is never shipped with the
project; point the configuration at it:

    JEV_CHECKPOINT=C:/Users/<you>/Downloads/best_model.pt   # the fine-tuned weights
    JEV_BASE_MODEL_DIR=C:/path/to/jev_ft/model_orig        # config + tokenizer of the base model
    JEV_DEVICE=cpu | cuda                                   # optional
    JEV_ADAPTER=C:/path/to/my_jev_adapter.py                # optional, see below

Faithful scoring needs the SAME renderer that was used in training
(``scripts/train_jev.py`` / ``eval_jev_finetuned.py``). Two ways to get it:

1. Recommended: write a small adapter file and set ``JEV_ADAPTER``. It must
   define ``load(checkpoint_path, base_model_dir, device)`` returning an object
   with ``scores(text) -> dict[label, float]`` for the six labels. Copy the
   renderer and verdict-slot code from your eval script into it; see
   ``jev_adapter_example.py`` at the project root.
2. Otherwise the built-in renderer below is used. It follows the description
   of the method (question + options, one verdict slot per option, yes/no
   embedding difference), but if its template differs from the training
   template the scores will be wrong. Run ``python evaluate_intent.py`` and
   compare with your 81.8 % test accuracy before trusting it.

If the checkpoint or base model is missing, loading raises ``JEVError``: the
pipeline then blocks with ``final_status=ERROR`` and ``jev_failed`` instead of
silently treating prompts as safe.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from pathlib import Path
from typing import Any

from core.layers.contracts import INTENT_LABELS

logger = logging.getLogger(__name__)

CLASS_DESCRIPTIONS = {
    "safe": "Safe",
    "prompt_injection": "Prompt injection",
    "jailbreak": "Jailbreak",
    "harmful_instructions": "Harmful instructions",
    "manipulation": "Manipulation",
    "self_harm_risk": "Self-harm risk",
}
QUESTION = "Classify the user's message into exactly one safety category."
VERDICT_SLOT = "<verdict>"


class FineTunedJEVError(RuntimeError):
    pass


def _load_adapter(path: str):
    spec = importlib.util.spec_from_file_location("guardgpt_jev_adapter", path)
    if spec is None or spec.loader is None:
        raise FineTunedJEVError(f"cannot import JEV adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load"):
        raise FineTunedJEVError("JEV adapter must define load(checkpoint_path, base_model_dir, device)")
    return module


class _BuiltinScorer:
    """Best-effort renderer + yes/no verdict scoring with transformers."""

    def __init__(self, checkpoint: Path, base_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
        model = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=dtype)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise FineTunedJEVError("checkpoint is not a state dict")
        own = model.state_dict()
        if not any(key in own for key in state):
            # Common wrapper prefixes from training scripts.
            for prefix in ("model.", "module.", "backbone.", "lm."):
                stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                if any(key in own for key in stripped):
                    state = stripped
                    break
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(own) - len(missing)
        if loaded < 0.9 * len(own):
            raise FineTunedJEVError(
                f"checkpoint matched only {loaded}/{len(own)} base-model tensors; "
                "use JEV_ADAPTER with your training loader"
            )
        if missing or unexpected:
            logger.warning("JEV checkpoint: %d missing, %d unexpected tensors", len(missing), len(unexpected))
        self.model = model.to(device).eval()
        yes = self.tokenizer.encode(" yes", add_special_tokens=False)[-1]
        no = self.tokenizer.encode(" no", add_special_tokens=False)[-1]
        embeddings = self.model.get_output_embeddings().weight
        self.direction = (embeddings[yes] - embeddings[no]).float()

    def render(self, text: str) -> tuple[str, list[int]]:
        lines = [QUESTION, f"Message: {text}", "Options:"]
        for label in INTENT_LABELS:
            lines.append(f"- {CLASS_DESCRIPTIONS[label]}: {VERDICT_SLOT}")
        return "\n".join(lines), []

    def scores(self, text: str) -> dict[str, float]:
        torch = self.torch
        rendered, _ = self.render(text)
        pieces = rendered.split(VERDICT_SLOT)
        ids: list[int] = []
        slots: list[int] = []
        for index, piece in enumerate(pieces):
            ids.extend(self.tokenizer.encode(piece, add_special_tokens=False))
            if index < len(INTENT_LABELS):
                slots.append(len(ids) - 1)  # hidden state predicting the verdict token
        with torch.no_grad():
            output = self.model(torch.tensor([ids], device=self.device), output_hidden_states=True)
            hidden = output.hidden_states[-1][0].float()
            values = hidden[slots] @ self.direction.to(hidden.device)
        return {label: float(value) for label, value in zip(INTENT_LABELS, values)}


class FineTunedJEV:
    """Lazy, thread-safe wrapper; ``classify`` returns label, confidence, scores."""

    def __init__(self, checkpoint: str | None = None, base_dir: str | None = None,
                 adapter: str | None = None, device: str | None = None) -> None:
        self.checkpoint = checkpoint or os.getenv("JEV_CHECKPOINT", "")
        self.base_dir = base_dir or os.getenv("JEV_BASE_MODEL_DIR", "")
        self.adapter = adapter or os.getenv("JEV_ADAPTER", "")
        self.device = device or os.getenv("JEV_DEVICE", "cpu")
        self._scorer: Any = None
        self._lock = threading.Lock()
        self.status: dict[str, Any] = {"loaded": False}

    def check_config(self) -> dict[str, Any]:
        problems = []
        if not self.checkpoint or not Path(self.checkpoint).is_file():
            problems.append(f"JEV_CHECKPOINT not found: {self.checkpoint or '(unset)'}")
        if not self.adapter and (not self.base_dir or not Path(self.base_dir).is_dir()):
            problems.append(f"JEV_BASE_MODEL_DIR not found: {self.base_dir or '(unset)'}")
        if self.adapter and not Path(self.adapter).is_file():
            problems.append(f"JEV_ADAPTER not found: {self.adapter}")
        return {"ok": not problems, "problems": problems, "checkpoint": self.checkpoint,
                "base_model_dir": self.base_dir, "adapter": self.adapter or "builtin renderer"}

    def _load(self):
        if self._scorer is not None:
            return self._scorer
        with self._lock:
            if self._scorer is None:
                config = self.check_config()
                if not config["ok"]:
                    raise FineTunedJEVError("; ".join(config["problems"]))
                if self.adapter:
                    module = _load_adapter(self.adapter)
                    self._scorer = module.load(self.checkpoint, self.base_dir, self.device)
                else:
                    logger.warning("Fine-tuned JEV uses the built-in renderer; set JEV_ADAPTER to "
                                   "use the renderer from your training script.")
                    self._scorer = _BuiltinScorer(Path(self.checkpoint), Path(self.base_dir), self.device)
                self.status = {"loaded": True, **config}
                logger.info("Fine-tuned JEV loaded from %s", self.checkpoint)
        return self._scorer

    def classify(self, text: str) -> tuple[str, float, dict[str, float]]:
        import math

        scores = self._load().scores(text)
        missing = [label for label in INTENT_LABELS if label not in scores]
        if missing:
            raise FineTunedJEVError(f"JEV adapter returned no score for {missing}")
        values = [float(scores[label]) for label in INTENT_LABELS]
        top = max(values)
        exps = [math.exp(v - top) for v in values]
        total = sum(exps)
        probabilities = {label: e / total for label, e in zip(INTENT_LABELS, exps)}
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label], {k: round(v, 4) for k, v in probabilities.items()}

"""Process-wide singletons shared by the MCP server's services.

One embedding model, one FAISS index and one trained intent head per process;
every layer and every MCP tool uses the same objects.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = [
    "dataset_loader", "embedding_model", "input_guard", "intent_layer",
    "jev_model", "policy_engine", "complete_pipeline", "reset_runtime",
]


@lru_cache(maxsize=1)
def dataset_loader():
    """Validated dataset + FAISS index, or None when it cannot be loaded."""
    from core.dataset_loader import DatasetLoader

    try:
        loader = DatasetLoader()
        loader.load()
        return loader
    except Exception as error:
        logger.warning("Dataset / FAISS evidence unavailable: %s", error)
        return None


@lru_cache(maxsize=1)
def embedding_model():
    """Base all-MiniLM-L6-v2 encoder (the FAISS index's vector space)."""
    loader = dataset_loader()
    if loader is not None and getattr(loader, "_model", None) is not None:
        return loader._model
    from core.dataset_loader import MODEL_NAME, _load_embedding_model

    try:
        return _load_embedding_model(MODEL_NAME)
    except Exception as error:
        logger.warning("Embedding model unavailable: %s", error)
        return None


@lru_cache(maxsize=1)
def input_guard():
    from core.layers.layer1_guard import InputGuard

    return InputGuard(dataset_provider=dataset_loader)


@lru_cache(maxsize=1)
def intent_layer():
    from core.layers.layer2_intent import IntentIntelligence

    return IntentIntelligence(encoder_provider=embedding_model, dataset_provider=dataset_loader)


@lru_cache(maxsize=1)
def jev_model():
    from core.layers.jev import JEVModel

    return JEVModel()


@lru_cache(maxsize=1)
def policy_engine():
    from core.layers.layer3_policy import PolicyEngine

    return PolicyEngine()


@lru_cache(maxsize=1)
def complete_pipeline():
    from core.complete_pipeline import CompletePipeline

    return CompletePipeline()


def reset_runtime() -> None:
    """Drop cached singletons (tests and config reloads)."""
    for factory in (dataset_loader, embedding_model, input_guard, intent_layer,
                    jev_model, policy_engine, complete_pipeline):
        factory.cache_clear()

"""Name -> model class registry, so the CLI and config can select a model by string."""
from __future__ import annotations

from typing import Callable

from .base import SentimentModel
from .textblob_model import TextBlobModel
from .vader_model import VaderModel
from .transformer_model import TransformerModel
from .pytorch_model import PyTorchModel

MODEL_REGISTRY: dict[str, Callable[[], SentimentModel]] = {
    "textblob": TextBlobModel,
    "vader": VaderModel,
    "pytorch": PyTorchModel,
    "transformer": TransformerModel,
}


def get_model(name: str) -> SentimentModel:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}") from exc
    return factory()

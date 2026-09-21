"""Hugging Face transformer sentiment model.

Uses a small pretrained model for *inference only* - no fine-tuning or
training happens here. Default: `cardiffnlp/twitter-roberta-base-sentiment-latest`,
a RoBERTa model fine-tuned by its authors on tweets for 3-class sentiment
(negative/neutral/positive), which is a reasonable match for short,
informal Reddit titles.

Because this model was fine-tuned on data similar to any Twitter/social
sentiment benchmark, comparing it against TextBlob/VADER on a dataset like
TweetEval is not an apples-to-apples "which algorithm is smarter" test -
this model has effectively seen data from the same distribution during its
own training. That caveat belongs in the README, not hidden here.
"""
from __future__ import annotations

import os

from .base import SentimentModel, SentimentPrediction

DEFAULT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Some checkpoints expose generic LABEL_0/1/2 heads instead of named ones;
# map defensively so this doesn't silently mis-map if the user points
# TRANSFORMER_MODEL_NAME at a different checkpoint.
_LABEL_ALIASES = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "neg": "negative",
    "pos": "positive",
    "neu": "neutral",
}


def _normalize_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    normalized = _LABEL_ALIASES.get(normalized, normalized)
    if normalized not in ("negative", "neutral", "positive"):
        raise ValueError(f"Unrecognized transformer label: {raw_label!r}")
    return normalized


class TransformerModel(SentimentModel):
    name = "transformer"

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or os.environ.get("TRANSFORMER_MODEL_NAME", DEFAULT_MODEL_NAME)
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - exercised only without the dependency
            raise ImportError(
                "TransformerModel requires 'transformers' and a backend "
                "('torch' by default). Install with: "
                "pip install -r requirements-transformer.txt"
            ) from exc
        # top_k=None returns scores for all classes instead of just the
        # argmax, which we want so we can build a signed polarity score.
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
        )
        self.model_name = model_name

    def predict(self, text: str) -> SentimentPrediction:
        text = text or ""
        if not text.strip():
            return SentimentPrediction(label="neutral", polarity=0.0, raw_scores={})

        # top_k=None returns a list of {label, score} dicts (one per class),
        # wrapped in an outer list because the pipeline supports batches.
        raw = self._pipeline(text, truncation=True)[0]
        scores = {_normalize_label(entry["label"]): float(entry["score"]) for entry in raw}
        top_label = max(scores, key=scores.get)

        # Build a signed polarity in [-1, 1] for the color mapping: positive
        # confidence contributes +score, negative contributes -score,
        # neutral contributes 0. This is a simple, documented convention,
        # not a property of the model itself.
        polarity = scores.get("positive", 0.0) - scores.get("negative", 0.0)
        polarity = max(-1.0, min(1.0, polarity))

        return SentimentPrediction(label=top_label, polarity=polarity, raw_scores=scores)

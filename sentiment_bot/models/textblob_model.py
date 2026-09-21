"""TextBlob sentiment model - the original 2024 baseline, unchanged.

`analyze_sentiment()` from the original notebook is preserved here exactly
(TextBlob(text).sentiment.polarity); it's now just wrapped behind the common
`SentimentModel` interface so it can sit next to VADER and the transformer
in the evaluator.
"""
from __future__ import annotations

from .base import SentimentModel, SentimentPrediction, polarity_to_label


class TextBlobModel(SentimentModel):
    name = "textblob"

    def __init__(self) -> None:
        try:
            from textblob import TextBlob  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the dependency
            raise ImportError(
                "TextBlobModel requires the 'textblob' package. "
                "Install it with: pip install textblob"
            ) from exc
        self._TextBlob = TextBlob

    def predict(self, text: str) -> SentimentPrediction:
        text = text or ""
        polarity = float(self._TextBlob(text).sentiment.polarity) if text.strip() else 0.0
        label = polarity_to_label(polarity)
        return SentimentPrediction(label=label, polarity=polarity)

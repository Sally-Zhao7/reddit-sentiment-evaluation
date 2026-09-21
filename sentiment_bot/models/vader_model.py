"""VADER sentiment model - second baseline.

VADER's `compound` score is already a signed value in [-1, 1], similar in
spirit to TextBlob's polarity, but computed from a lexicon tuned for
social-media text (it handles things like punctuation emphasis, negation,
and emoticons), which is a closer match to Reddit titles than TextBlob's
more general-purpose lexicon.
"""
from __future__ import annotations

from .base import SentimentModel, SentimentPrediction, polarity_to_label


class VaderModel(SentimentModel):
    name = "vader"

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:  # pragma: no cover - exercised only without the dependency
            raise ImportError(
                "VaderModel requires the 'vaderSentiment' package. "
                "Install it with: pip install vaderSentiment"
            ) from exc
        self._analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> SentimentPrediction:
        text = text or ""
        scores = self._analyzer.polarity_scores(text) if text.strip() else {"compound": 0.0}
        polarity = float(scores["compound"])
        label = polarity_to_label(polarity)
        return SentimentPrediction(label=label, polarity=polarity, raw_scores=scores)

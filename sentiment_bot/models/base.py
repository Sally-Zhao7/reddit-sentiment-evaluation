"""Common sentiment-model interface.

Every model (TextBlob, VADER, the Hugging Face transformer) implements this
same interface so the evaluator and the Reddit bot can use them
interchangeably, and so a new model can be added later without touching
either of those.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# The three-class label space every model must map its output onto. This is
# also the label space expected in the evaluation CSV's ground-truth column.
LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class SentimentPrediction:
    label: str  # one of LABELS
    polarity: float  # signed score in [-1, 1]; used for the color mapping
    raw_scores: Optional[dict] = field(default=None)  # optional per-class scores, if the model has them

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        if not -1.0 <= self.polarity <= 1.0:
            raise ValueError(f"polarity must be in [-1, 1], got {self.polarity!r}")


def polarity_to_label(polarity: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
    """Shared continuous-polarity -> 3-class thresholding.

    Used by both the TextBlob and VADER wrappers so their outputs are
    comparable to each other and to the transformer's native 3-class output.
    +/-0.05 is VADER's own published convention for compound score
    thresholds; we reuse it for TextBlob too for consistency rather than
    inventing a separate cutoff. This is a modeling choice, not a
    measurement - see README "Limitations".
    """
    if polarity > pos_threshold:
        return "positive"
    if polarity < neg_threshold:
        return "negative"
    return "neutral"


class SentimentModel(ABC):
    """Base class for a sentiment model usable by both the evaluator and the bot."""

    #: short machine-friendly identifier, e.g. "textblob", "vader", "transformer"
    name: str = "base"

    @abstractmethod
    def predict(self, text: str) -> SentimentPrediction:
        """Predict sentiment for a single piece of text.

        Implementations should treat empty/whitespace-only text as neutral
        rather than raising, since Reddit titles/bodies can legitimately be
        very short or empty.
        """
        raise NotImplementedError

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        """Default batch implementation: predict one at a time.

        Models that support real batched inference (e.g. the transformer)
        may override this for efficiency; this is not required for
        correctness, and the evaluator does not assume it's overridden.
        """
        return [self.predict(text) for text in texts]

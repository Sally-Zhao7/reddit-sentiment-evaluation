"""Wraps the trained PyTorch checkpoint (see sentiment_bot/torch_sentiment/)
behind the common SentimentModel interface, the same pattern used by
TransformerModel: load once in __init__, run inference in predict().
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import LABELS, SentimentModel, SentimentPrediction

DEFAULT_CHECKPOINT_PATH = "checkpoints/pytorch_sentiment.pt"


class PyTorchModel(SentimentModel):
    name = "pytorch"

    def __init__(self, checkpoint_path: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without the dependency
            raise ImportError(
                "PyTorchModel requires 'torch'. Install it with: pip install torch"
            ) from exc

        from ..torch_sentiment.data import Vocab, collate_batch, tokenize
        from ..torch_sentiment.model import SentimentClassifier

        checkpoint_path = checkpoint_path or os.environ.get("PYTORCH_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. Train it first with: "
                "python -m sentiment_bot.torch_sentiment.train --train data/tweeteval_train.csv "
                "--val data/tweeteval_val.csv"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._torch = torch
        self._tokenize = tokenize
        self._collate_batch = collate_batch
        self.vocab = Vocab.from_dict(checkpoint["vocab"])
        self.model = SentimentClassifier(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, text: str) -> SentimentPrediction:
        tokens = self._tokenize(text or "")
        token_ids = self.vocab.encode(tokens) or [self.vocab.unk_id]
        batch = self._collate_batch([(token_ids, 0)], pad_id=self.vocab.pad_id)
        input_ids, attention_mask, _ = batch

        with self._torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = self._torch.softmax(logits, dim=1)[0]

        label_idx = int(probs.argmax())
        label = LABELS[label_idx]
        scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        # Same signed-polarity convention as TransformerModel, for a consistent color mapping.
        polarity = scores["positive"] - scores["negative"]

        return SentimentPrediction(label=label, polarity=polarity, raw_scores=scores)

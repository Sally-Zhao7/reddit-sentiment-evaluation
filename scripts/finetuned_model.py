"""distilroberta-base fine-tuned by this project on TweetEval train.

Unlike `transformer` (cardiffnlp/twitter-roberta-base-sentiment-latest, used
off the shelf), this checkpoint is produced by scripts/finetune_transformer.py
from a general-domain base model, on the same 45,615-example train split used
for the custom PyTorch baseline. Inference reuses TransformerModel unchanged.
"""
from __future__ import annotations

import os

from .transformer_model import TransformerModel

DEFAULT_FINETUNED_PATH = "checkpoints/distilroberta-tweeteval"


class FinetunedTransformerModel(TransformerModel):
    name = "finetuned"

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name or os.environ.get("FINETUNED_MODEL_PATH", DEFAULT_FINETUNED_PATH))

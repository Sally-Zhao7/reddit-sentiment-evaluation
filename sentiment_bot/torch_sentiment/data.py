"""Vocabulary, tokenization, and the PyTorch Dataset/collate function.

Kept deliberately simple:
  - tokenization is just lowercasing + a regex word split, no external
    tokenizer library
  - the vocabulary is a plain dict[str, int] built by counting words

The vocabulary MUST be built from training data only (`Vocab.build` is only
ever called on the train split in train.py) - using words seen in
validation/test to build the vocabulary would leak information about those
splits into the model before it's ever evaluated on them.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import Dataset

from ..models.base import LABELS

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase + extract word-like tokens. No stemming, no stopword removal."""
    return _TOKEN_PATTERN.findall((text or "").lower())


class Vocab:
    """Maps tokens to integer ids. Index 0 is always <pad>, index 1 is <unk>."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, max_size: int | None = None) -> "Vocab":
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text))

        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        # Most common first, so a small max_size keeps the most useful words.
        for token, freq in counts.most_common():
            if freq < min_freq:
                continue
            if max_size is not None and len(token_to_id) >= max_size:
                break
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str]) -> list[int]:
        """Unknown tokens map to <unk> rather than raising."""
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def to_dict(self) -> dict[str, int]:
        """Plain dict, safe to put inside a torch checkpoint."""
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, token_to_id: dict[str, int]) -> "Vocab":
        return cls(dict(token_to_id))


class TweetSentimentDataset(Dataset):
    """Wraps a list of (text, label) rows as encoded, unpadded id sequences.

    Padding happens later, per-batch, in `collate_batch` - that way a short
    example in a batch of long ones isn't padded more than it needs to be
    across the whole dataset.
    """

    def __init__(self, rows: list[tuple[str, str]], vocab: Vocab):
        self.vocab = vocab
        self.examples: list[tuple[list[int], int]] = []
        for text, label in rows:
            token_ids = vocab.encode(tokenize(text))
            if not token_ids:  # empty text after tokenization: treat as a single <unk>
                token_ids = [vocab.unk_id]
            self.examples.append((token_ids, LABEL_TO_ID[label]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.examples[idx]


def collate_batch(batch: list[tuple[list[int], int]], pad_id: int):
    """Pad a batch of variable-length id sequences to the longest one in the batch.

    Returns (token_ids [B, T], attention_mask [B, T], labels [B]). The mask
    is 1 for real tokens and 0 for padding, and is what the model's masked
    mean pooling uses so padding doesn't dilute the sentence representation.
    """
    max_len = max(len(ids) for ids, _ in batch)

    token_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)

    for i, (ids, label) in enumerate(batch):
        token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        labels[i] = label

    return token_ids, attention_mask, labels

"""The PyTorch model architecture.

token ids -> Embedding -> masked mean pooling -> Linear -> ReLU -> Linear -> 3 class logits

Intentionally simple: no RNN, no attention, no custom Transformer. Masked
mean pooling just averages the embeddings of the real (non-padding) tokens
in a sentence into one fixed-size vector, which a small feed-forward head
then classifies.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = 3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """token_ids, attention_mask: [batch_size, seq_len]. Returns logits: [batch_size, num_classes]."""
        embedded = self.embedding(token_ids)  # [B, T, E]

        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (embedded * mask).sum(dim=1)  # [B, E], padding contributes 0
        real_token_counts = mask.sum(dim=1).clamp(min=1)  # [B, 1], avoid divide-by-zero
        pooled = summed / real_token_counts  # [B, E], mean over real tokens only

        hidden = self.relu(self.fc1(pooled))  # [B, hidden_dim]
        logits = self.fc2(hidden)  # [B, num_classes]
        return logits

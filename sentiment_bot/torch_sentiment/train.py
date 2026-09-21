"""Train the PyTorch sentiment classifier.

Standard supervised training loop: build a vocabulary from the training
data, wrap train/val CSVs as DataLoaders, train with CrossEntropyLoss +
Adam, run validation after every epoch, and save a checkpoint at the end.

IMPORTANT: only ever pass the TweetEval TRAIN split as --train and the
VAL split as --val. The TEST split is held out for evaluation.py /
evaluation/runner.py and must never be used here.

Usage:
    python -m sentiment_bot.torch_sentiment.train \\
        --train data/tweeteval_train.csv \\
        --val data/tweeteval_val.csv \\
        --out checkpoints/pytorch_sentiment.pt
"""
from __future__ import annotations

import argparse
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.runner import load_labeled_csv
from .data import TweetSentimentDataset, Vocab, collate_batch
from .model import SentimentClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, float]:
    """One pass over `loader`. Trains if `optimizer` is given, otherwise just evaluates.

    Returns (average_loss, accuracy).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for token_ids, attention_mask, labels in loader:
            logits = model(token_ids, attention_mask)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def train(
    train_path: Path,
    val_path: Path,
    out_path: Path,
    epochs: int = 5,
    batch_size: int = 32,
    embed_dim: int = 64,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    min_freq: int = 2,
    max_vocab_size: int = 20000,
) -> None:
    set_seed(seed)

    train_rows = load_labeled_csv(train_path)
    val_rows = load_labeled_csv(val_path)

    # Vocabulary is built from the TRAINING split only.
    vocab = Vocab.build((text for text, _ in train_rows), min_freq=min_freq, max_size=max_vocab_size)
    print(f"[+] vocab size: {len(vocab)} (from {len(train_rows)} training examples)")

    collate = partial(collate_batch, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        TweetSentimentDataset(train_rows, vocab), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        TweetSentimentDataset(val_rows, vocab), batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = SentimentClassifier(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim, pad_idx=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab.to_dict(),
            "config": {
                "vocab_size": len(vocab),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "pad_idx": vocab.pad_id,
            },
        },
        out_path,
    )
    print(f"[+] saved checkpoint to {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, type=Path, help="TweetEval TRAIN split CSV (text,label)")
    parser.add_argument("--val", required=True, type=Path, help="TweetEval VAL split CSV (text,label)")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/pytorch_sentiment.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    train(
        args.train, args.val, args.out,
        epochs=args.epochs, batch_size=args.batch_size,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        lr=args.lr, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

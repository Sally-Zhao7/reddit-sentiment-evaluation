"""Evaluation metrics for a single model's predictions against ground truth.

Uses scikit-learn's implementations rather than hand-rolling precision/
recall/F1 - there's no engineering value in reimplementing well-tested
metric code, and using a standard library makes the numbers easy to sanity
check independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..models.base import LABELS


@dataclass
class ModelMetrics:
    model_name: str
    n_examples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: list  # list of lists, rows/cols ordered as LABELS
    avg_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_examples": self.n_examples,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": list(LABELS),
            "avg_latency_ms": self.avg_latency_ms,
        }


def compute_metrics(
    model_name: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    latencies_ms: Sequence[float],
) -> ModelMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must be the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty dataset")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    return ModelMetrics(
        model_name=model_name,
        n_examples=len(y_true),
        accuracy=float(accuracy),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        confusion_matrix=cm.tolist(),
        avg_latency_ms=float(avg_latency),
    )


def format_comparison_table(all_metrics: list[ModelMetrics]) -> str:
    """Render a plain-text comparison table across models (also used in the README)."""
    headers = ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Avg latency (ms)", "N"]
    rows = [
        [
            m.model_name,
            f"{m.accuracy:.3f}",
            f"{m.macro_precision:.3f}",
            f"{m.macro_recall:.3f}",
            f"{m.macro_f1:.3f}",
            f"{m.avg_latency_ms:.2f}",
            str(m.n_examples),
        ]
        for m in all_metrics
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(len(headers))]
    lines = []
    lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)

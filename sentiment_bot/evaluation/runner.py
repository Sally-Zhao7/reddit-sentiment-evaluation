"""Evaluation CLI: run one or more sentiment models against a labeled CSV
and report accuracy / precision / recall / macro F1 / confusion matrix /
average latency, plus a cross-model comparison table.

This script produces numbers ONLY by actually running the models on the
given data - there is no code path that fabricates or hardcodes metrics.
If a model fails to load (e.g. a missing optional dependency), that model
is skipped with a clear message instead of silently omitted.

Usage:
    python -m sentiment_bot.evaluation.runner --data data/smoke_test_sample.csv
    python -m sentiment_bot.evaluation.runner --data path/to/labeled.csv --models textblob vader
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..models.base import LABELS, SentimentModel
from ..models.registry import MODEL_REGISTRY, get_model
from .metrics import ModelMetrics, compute_metrics, format_comparison_table


def load_labeled_csv(path: Path) -> list[tuple[str, str]]:
    """Load a CSV with columns `text,label`. label must be one of LABELS."""
    rows: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{path} must have a header with 'text' and 'label' columns, "
                f"got: {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):  # start=2: header is line 1
            text = row["text"]
            label = row["label"].strip().lower()
            if label not in LABELS:
                raise ValueError(
                    f"{path}:{i}: label {row['label']!r} is not one of {LABELS}"
                )
            rows.append((text, label))
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def evaluate_model(model: SentimentModel, rows: list[tuple[str, str]]) -> tuple[ModelMetrics, list[dict]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []
    per_example: list[dict] = []

    for text, true_label in rows:
        start = time.perf_counter()
        prediction = model.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        y_true.append(true_label)
        y_pred.append(prediction.label)
        latencies_ms.append(elapsed_ms)
        per_example.append(
            {
                "text_length": len(text),
                "true_label": true_label,
                "predicted_label": prediction.label,
                "polarity": prediction.polarity,
                "latency_ms": elapsed_ms,
                "correct": prediction.label == true_label,
            }
        )

    metrics = compute_metrics(model.name, y_true, y_pred, latencies_ms)
    return metrics, per_example


def run(data_path: Path, model_names: list[str], output_dir: Path) -> int:
    rows = load_labeled_csv(data_path)

    all_metrics: list[ModelMetrics] = []
    skipped: list[dict] = []
    per_model_examples: dict[str, list[dict]] = {}

    for name in model_names:
        print(f"[+] loading model: {name}", file=sys.stderr)
        try:
            model = get_model(name)
        except Exception as exc:  # missing optional dependency, bad config, etc.
            print(f"[!] skipping {name}: {exc}", file=sys.stderr)
            skipped.append({"model": name, "reason": str(exc)})
            continue

        print(f"[+] evaluating {name} on {len(rows)} examples...", file=sys.stderr)
        metrics, examples = evaluate_model(model, rows)
        all_metrics.append(metrics)
        per_model_examples[name] = examples
        print(
            f"[+] {name}: accuracy={metrics.accuracy:.3f} macro_f1={metrics.macro_f1:.3f} "
            f"avg_latency={metrics.avg_latency_ms:.2f}ms",
            file=sys.stderr,
        )

    if not all_metrics:
        print("[!] No model produced results (all were skipped). Nothing to report.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Just enough metadata to know what produced these numbers: which
    # dataset, how many rows, which models ran or were skipped, and when.
    run_metadata = {
        "timestamp_utc": timestamp,
        "dataset_path": str(data_path),
        "n_examples": len(rows),
        "models_requested": model_names,
        "models_skipped": skipped,
    }

    result_payload = {
        "run_metadata": run_metadata,
        "metrics": [m.to_dict() for m in all_metrics],
    }

    result_path = output_dir / f"comparison_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    table = format_comparison_table(all_metrics)
    print("\n" + table)
    print(f"\n[+] Wrote full results (including per-example predictions metadata) to {result_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Path to a labeled CSV with text,label columns")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to evaluate (default: all registered models)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Where to write result JSON")
    args = parser.parse_args(argv)

    return run(args.data, args.models, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

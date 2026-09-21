"""Download and convert the TweetEval sentiment dataset into the CSV schema
the evaluator expects (text,label with label in {negative,neutral,positive}).

TweetEval (Barbieri et al., 2020) is a public benchmark of short, informal,
human-annotated social-media text - a closer proxy for Reddit titles than
most sentiment datasets, and NOT something generated for this project.
Source: https://github.com/cardiffnlp/tweeteval

This script requires normal internet access. It could not be run inside
the sandboxed environment this project was originally built in (see the
top-level README's "What I could not verify here" section) - run it
yourself, then point the evaluator at its output.

Usage:
    python scripts/prepare_tweet_eval.py --split test --limit 300 --out data/tweeteval_test.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment"


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _load_label_mapping() -> dict[int, str]:
    """TweetEval's mapping.txt is authoritative for which integer means what -
    don't hardcode label order here in case it ever changes."""
    raw = _fetch_text(f"{BASE_URL}/mapping.txt")
    mapping: dict[int, str] = {}
    for line in raw.strip().splitlines():
        idx_str, name = line.strip().split("\t")
        mapping[int(idx_str)] = name.strip().lower()
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path (default: data/tweeteval_<split>.csv)")
    args = parser.parse_args(argv)

    out_path = args.out or Path("data") / f"tweeteval_{args.split}.csv"

    print(f"[+] fetching label mapping from {BASE_URL}/mapping.txt", file=sys.stderr)
    mapping = _load_label_mapping()

    text_url = f"{BASE_URL}/{args.split}_text.txt"
    label_url = f"{BASE_URL}/{args.split}_labels.txt"
    print(f"[+] fetching {text_url}", file=sys.stderr)
    texts = _fetch_text(text_url).strip("\n").split("\n")
    print(f"[+] fetching {label_url}", file=sys.stderr)
    label_ids = _fetch_text(label_url).strip("\n").split("\n")

    if len(texts) != len(label_ids):
        raise RuntimeError(f"Mismatched lengths: {len(texts)} texts vs {len(label_ids)} labels")

    rows = list(zip(texts, label_ids))
    if args.limit:
        rows = rows[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label_id in rows:
            writer.writerow([text, mapping[int(label_id)]])

    print(f"[+] wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

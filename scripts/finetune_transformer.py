"""Fine-tune distilroberta-base on the TweetEval sentiment train split.

Designed to run on a free Colab T4 (~15-25 min for 2 epochs). Uses the same
CSVs produced by scripts/prepare_tweet_eval.py, so the fine-tuned model sees
exactly the same 45,615 training examples as the custom PyTorch baseline.
The test split is NOT touched here; final numbers come from
sentiment_bot.evaluation.runner, like every other model.

Colab usage:
    !git clone https://github.com/Sally-Zhao7/reddit-sentiment-evaluation.git
    %cd reddit-sentiment-evaluation
    !pip install -q -r requirements.txt -r requirements-transformer.txt datasets accelerate scikit-learn
    !python scripts/prepare_tweet_eval.py --split train --out data/tweeteval_train.csv
    !python scripts/prepare_tweet_eval.py --split val --out data/tweeteval_val.csv
    !python scripts/finetune_transformer.py \
        --train data/tweeteval_train.csv --val data/tweeteval_val.csv \
        --out checkpoints/distilroberta-tweeteval
    # then download: !zip -r ft.zip checkpoints/distilroberta-tweeteval
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_split(path: str, tokenizer, max_len: int) -> Dataset:
    df = pd.read_csv(path).dropna(subset=["text"])
    df["labels"] = df["label"].str.strip().str.lower().map(LABEL2ID)
    if df["labels"].isna().any():
        raise ValueError(f"{path} has labels outside {LABELS}")
    ds = Dataset.from_pandas(df[["text", "labels"]].astype({"labels": int}), preserve_index=False)
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    y = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        # Macro-averaged recall is the official TweetEval sentiment metric.
        "macro_recall": recall_score(y, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default="distilroberta-base")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,  # so the HF pipeline emits "negative"/"neutral"/"positive"
        label2id=LABEL2ID,
    )

    train_ds = load_split(args.train, tokenizer, args.max_len)
    val_ds = load_split(args.val, tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=f"{args.out}-runs",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_recall",  # model selection on val only
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Best validation metrics:", trainer.evaluate())

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()

# Reddit Sentiment Bot: Model Evaluation & Automation

A Reddit bot that reads a submission title, scores its sentiment, and
replies with a color suggestion, plus a small evaluation harness for
comparing sentiment models against each other on the same data.

This started as a 2024 course project (a single notebook using TextBlob).
This repo is a 2026 extension of that project into a modular, testable
Python implementation focused on reproducible model evaluation and reliable
API automation, then extended further with a small trainable PyTorch
sentiment classifier as a fourth model to compare.

## Table of contents

- [Original 2024 project](#original-2024-project)
- [2026 extension](#2026-extension)
- [The four sentiment approaches](#the-four-sentiment-approaches)
- [Architecture](#architecture)
- [Dataset and evaluation methodology](#dataset-and-evaluation-methodology)
- [Directory structure](#directory-structure)
- [Setup](#setup)
- [Training the PyTorch model](#training-the-pytorch-model)
- [Running the evaluation](#running-the-evaluation)
- [Running the Reddit bot](#running-the-reddit-bot)
- [Testing](#testing)
- [Results](#results)
- [Limitations](#limitations)
- [What I could not verify in the build environment](#what-i-could-not-verify-in-the-build-environment)
- [Component guide (for interview prep)](#component-guide-for-interview-prep)

## Original 2024 project

`legacy/original_2024_bot.py` is a verbatim transcription of the notebook
submitted in May 2024 (see `Sirou_Zhao_Reddit_Sentiment_Bot.pdf` for the
original writeup). It:

- uses PRAW to stream new submissions from `r/colors`
- uses TextBlob to compute a sentiment **polarity** score in `[-1, 1]`
- maps that score into one of 7 colors via fixed thresholds
- replies `"How about the color {color}?"` to the submission
- checks whether it already replied by searching comments for that phrase
- retries after a fixed 60-second sleep on any exception

It worked, but had real problems, visible in the notebook's own captured
output:

```
API error: RATELIMIT: "Looks like you've been doing that a lot. Take a break for
9 minutes before trying again." on field 'ratelimit'
```

A fixed 60-second sleep doesn't help when Reddit is explicitly asking for a
9-minute break - the next attempt just gets rate-limited again. There were
also correctness issues: `already_replied` matched the reply text against
**any** commenter, not just the bot itself, so a human joking "how about the
color blue?" in a thread would have permanently silenced the bot there.

## 2026 extension

Two things motivated revisiting this project:

1. **Model evaluation, not just automation.** The original only ever used
   TextBlob and never measured how good it actually was.
2. **Reliability.** The rate-limit and duplicate-detection bugs above
   matter more once a project is meant to run unattended.

The 2026 work added, in order:

- a common `SentimentModel` interface so models are interchangeable
- two more lexicon-based baselines (VADER) and a pretrained transformer
- a reproducible evaluation pipeline (accuracy / precision / recall / macro
  F1 / confusion matrix / latency, computed with scikit-learn)
- rate-limit-aware retry with backoff, replacing the fixed 60s sleep
- a fix for the duplicate-reply bug
- tests
- a small, trainable PyTorch sentiment classifier, added as a fourth model
  once the evaluation pipeline above already existed to compare it against

The custom PyTorch classifier is intentionally simple - an embedding layer,
masked mean pooling, and a two-layer feed-forward head - and is meant as a
**trainable neural baseline** to sit alongside the lexicon-based and
pretrained approaches, not as a demonstration of a more advanced
architecture.

## The four sentiment approaches

All four implement the same interface (`sentiment_bot/models/base.py`) and
map onto the same 3-class label space (`negative` / `neutral` / `positive`).

| Model | What it is | Notes |
|---|---|---|
| `textblob` | The original 2024 baseline | Lexicon-based; `TextBlob(text).sentiment.polarity`, thresholded into 3 classes at ±0.05 |
| `vader` | Second lexicon-based baseline | Tuned for social-media text (negation, emphasis, emoticons); uses its own `compound` score, same ±0.05 thresholding |
| `pytorch` | Custom, trained from scratch here | Embedding -> masked mean pooling -> Linear -> ReLU -> Linear, trained on TweetEval's training split |
| `transformer` | Pretrained Hugging Face model | `cardiffnlp/twitter-roberta-base-sentiment-latest`, **inference only, not fine-tuned here** - natively outputs negative/neutral/positive |

**Comparability caveats:**
- The transformer checkpoint was fine-tuned by its authors on data very
  similar to TweetEval (short, informal social-media text), so it has an
  inherent home-field advantage evaluated on TweetEval.
- The PyTorch model is trained from scratch on TweetEval's own training
  split, so it's fitted specifically to this data distribution too - its
  numbers say more about how well a small model can fit this dataset than
  about general-purpose sentiment understanding.
- TextBlob and VADER are unsupervised (no training data at all), so this
  isn't a fully controlled comparison across all four - it's four
  different, reasonable ways to build a sentiment classifier, evaluated the
  same way.

## Architecture

```
Reddit submission title
        │
        ▼
 SentimentModel.predict(text)  ──►  SentimentPrediction(label, polarity)
        │                                  │
        │                                  ▼
        │                         get_color_by_mood(polarity)  (unchanged from 2024)
        ▼                                  │
 evaluation/runner.py                      ▼
 (offline, labeled CSV in)          reddit/bot.py
 (metrics + comparison table out)   (online, live/dry-run Reddit replies)
```

Both the offline evaluator and the online bot are built on the same
`SentimentModel` interface, so adding a model means writing one class, not
touching the bot or the evaluator, and the exact prediction code path that
gets measured in evaluation is the one that runs against live Reddit data.

## Dataset and evaluation methodology

Evaluation data is a CSV with `text,label` columns (label one of
`negative`/`neutral`/`positive`). Two sources are used, for different
purposes:

- **`data/smoke_test_sample.csv`** - a small synthetic dataset (12 short,
  clearly-polarized sentences) generated during this refactor, used only to
  verify the evaluation pipeline runs end to end. It is **not** used as a
  benchmark or for reported model performance.
- **[TweetEval](https://github.com/cardiffnlp/tweeteval) sentiment task** -
  a public, human-annotated dataset of short social-media text, downloaded
  via `scripts/prepare_tweet_eval.py`. This is the actual evaluation data.

**Split usage (no data leakage):**

| Split | Used for |
|---|---|
| `train` | Building the PyTorch model's vocabulary and training its weights. Nothing else touches this split. |
| `val` | Monitoring the PyTorch model during training (printed after every epoch); not used to update weights. |
| `test` | **Held out.** Used only for the final evaluation/comparison across all four models, after training is finished. |

The PyTorch model is never trained or vocabulary-built on the validation or
test split, and the test split is never used during training or model
selection - only in `evaluation/runner.py`, at the end.

**Metrics** (via scikit-learn): accuracy, macro precision/recall/F1,
confusion matrix, and average per-example inference latency, computed by
`sentiment_bot/evaluation/metrics.py` from predictions the evaluator
actually produced. No metric in this repo is hand-typed.

## Directory structure

```
reddit-sentiment-eval/
├── README.md
├── requirements.txt                 # bot + baseline models + PyTorch + tests
├── requirements-transformer.txt     # extra: transformers (needs torch, already in requirements.txt)
├── .gitignore
├── .env.example
├── pytest.ini
├── legacy/
│   └── original_2024_bot.py         # verbatim 2024 notebook code, for reference only
├── data/
│   ├── README.md
│   └── smoke_test_sample.csv        # synthetic, pipeline smoke test only
├── scripts/
│   └── prepare_tweet_eval.py        # downloads + converts the public TweetEval dataset
├── checkpoints/                     # trained PyTorch model, gitignored except .gitkeep
├── results/                         # evaluation run output (JSON), gitignored except .gitkeep
├── sentiment_bot/
│   ├── config.py                    # env-var based configuration
│   ├── colors.py                    # polarity -> color mapping (unchanged from 2024)
│   ├── utils.py                     # logging setup + retry/backoff helper
│   ├── models/
│   │   ├── base.py                  # SentimentModel ABC, SentimentPrediction, shared thresholding
│   │   ├── textblob_model.py
│   │   ├── vader_model.py
│   │   ├── pytorch_model.py         # wraps a trained torch_sentiment checkpoint
│   │   ├── transformer_model.py
│   │   └── registry.py              # name -> model class lookup
│   ├── torch_sentiment/             # the custom PyTorch model
│   │   ├── data.py                  # vocabulary, tokenization, Dataset, padding/collate
│   │   ├── model.py                 # the nn.Module (embedding + pooling + FFN)
│   │   └── train.py                 # training loop CLI (train/val, checkpoint save)
│   ├── evaluation/
│   │   ├── metrics.py               # accuracy/P/R/F1/confusion matrix/latency
│   │   └── runner.py                # CLI: labeled CSV in, comparison table + JSON out
│   └── reddit/
│       ├── client.py                # praw.Reddit construction
│       ├── replies.py               # duplicate-detection / reply-text logic (no praw import)
│       └── bot.py                   # streaming loop + CLI
└── tests/
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# Optional, only needed for the pretrained transformer model:
pip install -r requirements-transformer.txt

cp .env.example .env
# edit .env with real Reddit app credentials if you intend to run the bot
```

## Training the PyTorch model

```bash
# 1. Download the official TweetEval train/val/test splits
python scripts/prepare_tweet_eval.py --split train --out data/tweeteval_train.csv
python scripts/prepare_tweet_eval.py --split val   --out data/tweeteval_val.csv
python scripts/prepare_tweet_eval.py --split test  --out data/tweeteval_test.csv

# 2. Train (uses train + val only; never touches test)
python -m sentiment_bot.torch_sentiment.train \
    --train data/tweeteval_train.csv \
    --val data/tweeteval_val.csv \
    --out checkpoints/pytorch_sentiment.pt
```

This prints training/validation loss and accuracy after every epoch and
saves one checkpoint (model weights + vocabulary + architecture config) to
`--out`. The evaluator (`pytorch` model) loads that checkpoint at
`checkpoints/pytorch_sentiment.pt` by default, or from
`PYTORCH_CHECKPOINT_PATH` if set.

## Running the evaluation

```bash
# Pipeline smoke test (synthetic, not a benchmark - see data/README.md)
python -m sentiment_bot.evaluation.runner --data data/smoke_test_sample.csv

# Real evaluation, on the held-out TweetEval test split
python -m sentiment_bot.evaluation.runner --data data/tweeteval_test.csv

# Only specific models
python -m sentiment_bot.evaluation.runner --data data/tweeteval_test.csv --models textblob vader pytorch
```

Each run prints a comparison table to stdout and writes a JSON report
(per-model metrics + run metadata: dataset path, timestamp, models
requested/skipped) to `results/comparison_<timestamp>.json`. If a model's
dependency or checkpoint isn't available, it's skipped with an explicit
message rather than silently omitted or reported as zero.

## Running the Reddit bot

```bash
# Dry run (default): logs what it WOULD reply, never calls submission.reply()
python -m sentiment_bot.reddit.bot

# Live: actually posts replies to Reddit
python -m sentiment_bot.reddit.bot --live

# Override the subreddit or model for one run
python -m sentiment_bot.reddit.bot --subreddit colors --model vader
```

Requires `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`,
`REDDIT_PASSWORD` in `.env` (see `.env.example`) - create a "script" app at
https://www.reddit.com/prefs/apps to get these. The bot defaults to
**dry run** so a fresh checkout never posts to Reddit by accident; `--live`
or `BOT_DRY_RUN=false` opts in.

## Testing

```bash
pytest
# or, with no dependencies at all beyond the standard library:
python -m unittest discover -s tests
```

Model-specific tests (TextBlob/VADER/transformer/PyTorch) are automatically
skipped when that library isn't installed, rather than failing.

## Results

*(Intentionally left blank.)* This section only gets filled in once the
PyTorch model has actually been trained and all four models have actually
been run against the held-out TweetEval test split - see "What I could not
verify" below. Once run, this section should look like:

```
Model       | Accuracy | Macro P | Macro R | Macro F1 | Avg latency (ms) | N
------------+----------+---------+---------+----------+------------------+-----
textblob    | ?        | ?       | ?       | ?        | ?                | ?
vader       | ?        | ?       | ?       | ?        | ?                | ?
pytorch     | ?        | ?       | ?       | ?        | ?                | ?
transformer | ?        | ?       | ?       | ?        | ?                | ?
```
filled in from the actual `results/comparison_*.json` produced by a real
run - never typed in by hand.

## Limitations

- **Thresholding is a modeling choice, not a measurement.** TextBlob and
  VADER produce continuous scores; converting them to 3 classes at ±0.05
  is a defensible but arbitrary choice, and it directly affects reported
  accuracy/F1.
- **The transformer and the PyTorch model both have a data-distribution
  advantage on TweetEval** (see "Comparability caveats" above) - this is
  not a fully controlled comparison of the four approaches.
- **The PyTorch model is small and simple on purpose**: one embedding
  layer, mean pooling (no attention or recurrence), and a small
  feed-forward head. It's meant to demonstrate the core PyTorch training
  workflow clearly, not to be competitive with the transformer.
- **Latency numbers are single-example, single-machine, CPU (unless you
  have a GPU set up), batch size 1.**
- **`contains_keywords` is a no-op**, preserved from the original notebook.
  There's no real content filtering.
- **Privacy:** the bot still reads and reacts to public post titles without
  per-user consent, the same ethical tension the original notebook raised.
- **Not production-scale.** A single process streaming one subreddit,
  trained on a laptop-sized dataset. That's an intentional scope decision
  for a graduate-level portfolio project, not an oversight.

## What I could not verify in the build environment

This project was built inside a sandboxed environment with no access to
PyPI, apt, GitHub, or Hugging Face, and no `torch` installed (only pandas,
numpy, scikit-learn, requests, PyYAML, and python-dotenv were
pre-installed). As a direct result:

- I could **not** install `torch`, so I could not actually train the
  PyTorch model, run it, or produce any of its metrics.
- I could not download TweetEval, so `scripts/prepare_tweet_eval.py` was
  reviewed but never executed end to end.
- I could not run `textblob`, `vader`, or `transformer` predictions either,
  for the same reason as before - this repo still ships with **no
  evaluation numbers** for any model.
- I could not run the Reddit bot against live Reddit.

What I *did* verify, by actually running it in that sandbox:
- All modules (including `torch_sentiment/`'s data and model code, and
  `models/pytorch_model.py`) import without error, because every
  torch-dependent import is deferred to inside a function/`__init__`, not
  at module load time - the one exception is `torch_sentiment/train.py`,
  which imports `torch` at the top since it's a standalone training script
  never meant to run without it.
- All non-torch unit tests pass; every torch-dependent test is
  automatically skipped (not run, not passed) - see exact counts below.
- The evaluation CLI's argument parsing, CSV loading/validation, scoring,
  and JSON output, exercised end-to-end with a fake in-test model standing
  in for the real ones, and separately confirmed to fail gracefully (clear
  per-model skip messages, non-zero exit code, no fabricated numbers) when
  run for real against `data/smoke_test_sample.csv` with no model
  dependencies installed - now correctly including `pytorch` as a fourth
  model that's skipped for the same reason.
- The bot CLI's argument parsing and fail-fast behavior when credentials or
  model dependencies are missing.

**Before you rely on this for an interview, please run, on your own
machine with normal internet access:**
1. `pip install -r requirements.txt -r requirements-transformer.txt`
2. `pytest` - confirm all tests pass with zero skips this time
3. The three `scripts/prepare_tweet_eval.py` commands, then
   `python -m sentiment_bot.torch_sentiment.train ...` (see "Training the
   PyTorch model" above) - this is what actually produces a checkpoint
4. `python -m sentiment_bot.evaluation.runner --data data/tweeteval_test.csv`
   - this produces the real numbers for the Results section
5. If you want to test the bot live, set up a Reddit script app and run
   `python -m sentiment_bot.reddit.bot` (dry run first) against a
   low-traffic subreddit you control before ever using `--live` on
   `r/colors` or anywhere with real users.

## Component guide (for interview prep)

- **`colors.py`** - pure function, `polarity -> color name`, unchanged from
  2024. Simplest thing in the repo.
- **`models/base.py`** - the `SentimentModel` ABC and `SentimentPrediction`
  dataclass, which validates its own label/polarity on construction.
  `polarity_to_label` is the shared ±0.05 thresholding used by both
  lexicon-based models.
- **`models/textblob_model.py` / `vader_model.py` / `transformer_model.py`
  / `pytorch_model.py`** - one wrapper per model. Each lazily imports its
  third-party dependency (or, for `pytorch_model.py`, `torch` plus
  `torch_sentiment`) *inside* `__init__`, not at module level, so the rest
  of the package stays importable and testable even if that one dependency
  is missing.
- **`torch_sentiment/data.py`** - `Vocab.build` counts words in the
  training texts and assigns ids (index 0 is always `<pad>`, index 1
  `<unk>`); anything not seen during training maps to `<unk>` at
  prediction time. `collate_batch` pads a batch of variable-length token-id
  lists to the longest sequence in that batch and builds an attention mask
  (1 = real token, 0 = padding) that the model uses for pooling.
- **`torch_sentiment/model.py`** - `SentimentClassifier`: an
  `nn.Embedding` looks up a vector per token; multiplying by the attention
  mask and averaging gives one fixed-size vector per sentence (masked mean
  pooling) regardless of its length; a `Linear -> ReLU -> Linear` head then
  produces 3 class logits.
- **`torch_sentiment/train.py`** - standard supervised training loop:
  build the vocab from the train split only, wrap train/val as
  `DataLoader`s, then for each epoch: `model.train()`, iterate batches
  computing `CrossEntropyLoss`, `loss.backward()`, `optimizer.step()`
  (Adam); then `model.eval()` + `torch.no_grad()` over the validation set
  to report (not train on) validation loss/accuracy. Saves one checkpoint
  (weights + vocab + config) at the end.
- **`evaluation/metrics.py`** - a typed wrapper around `sklearn.metrics`
  plus a plain-text table formatter. Worth being able to explain macro vs.
  micro F1 (macro treats each class equally regardless of frequency).
- **`evaluation/runner.py`** - load CSV -> for each model, time each
  prediction and score against ground truth -> write a table and a JSON
  file with run metadata. Missing model dependencies/checkpoints are
  caught and reported per-model rather than crashing the whole run.
- **`reddit/replies.py`** - has zero praw imports; everything here operates
  on plain submission/comment objects, which is what makes it possible to
  unit-test with small fake classes instead of a mocked praw client.
  `already_replied_by_bot` is the fix for the original bug: it checks the
  comment's *author* against the bot's own username, not just the comment
  text.
- **`utils.py`** - two independent, deliberately small things:
  `configure_logging` (plain stdlib logging, one line per event) and
  `retry_call` (retries a function on transient errors, using the wait
  time Reddit's own rate-limit message names when there is one, otherwise
  exponential backoff with jitter). About 100 lines total, meant to be
  explainable in a couple of minutes.
- **`reddit/bot.py`** - `process_submission` handles one post: keyword
  filter (currently a no-op) -> duplicate check -> predict -> log -> reply
  (through `retry_call`) or log-only in dry-run mode. `run_bot` is the
  streaming loop, wrapping each submission's processing in a try/except so
  one bad submission can't kill the whole stream.
- **`config.py`** - all Reddit credentials and bot knobs come from
  environment variables, loaded via `python-dotenv` if a `.env` file is
  present. Nothing sensitive is imported from a Python source file, unlike
  the original notebook's `%run reddit_keys.py`.

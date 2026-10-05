# Reddit Sentiment Evaluation

![tests](https://github.com/Sally-Zhao7/reddit-sentiment-evaluation/actions/workflows/tests.yml/badge.svg)

An accuracy-vs-latency evaluation of sentiment models — lexicon baselines, a lightweight PyTorch classifier, and a DistilRoBERTa I fine-tuned — on the same held-out dataset, with a Reddit bot as the original application.

This began as a 2024 course project: a Reddit bot that used TextBlob sentiment polarity to recommend colors. In Sep 2026, I extended it into a reproducible model-evaluation pipeline with a trainable PyTorch baseline, a fine-tuned Transformer, additional sentiment models, testing, and more reliable Reddit API handling.

## Results

All models were evaluated on the same **12,284-example held-out TweetEval sentiment test split**. Macro recall is the official TweetEval sentiment metric.

| Model | Accuracy | Macro Recall | Macro F1 | CPU time / 1k texts |
|---|---:|---:|---:|---:|
| TextBlob | 0.487 | 0.490 | 0.464 | 0.15 s |
| VADER | 0.530 | 0.570 | 0.529 | 0.06 s |
| Custom PyTorch (lightweight) | 0.576 | 0.578 | 0.571 | 0.11 s |
| DistilRoBERTa, fine-tuned here | 0.702 | 0.723 | 0.703 | 27.0 s |
| Twitter-RoBERTa (off the shelf, reference) | 0.723 | 0.734 | 0.725 | — |

**Accuracy vs. latency.** Fine-tuning DistilRoBERTa on the same 45,615 training examples raises macro F1 from 0.571 to 0.703. The lightweight PyTorch model is about 13 points lower but **~240× faster** on CPU, which suits a bot that scores every incoming post on cheap hardware; the fine-tuned model is the choice when accuracy matters more than cost.

The off-the-shelf `cardiffnlp/twitter-roberta-base-sentiment-latest` scores highest, but it was pretrained on tweets and trained by its authors on TweetEval-style data, so it is a reference ceiling rather than a like-for-like comparison. Its scores come from an earlier run of the same evaluator; it was not re-timed on the machine used for the latency column.

Latency is per example with batch size 1 (how the bot processes one post at a time) on a single CPU, reported per 1,000 texts. Treat it as relative, not absolute: on a GPU or with batching, the Transformers would be much faster.

## Models

All models implement the same `SentimentModel` interface and output `negative`, `neutral`, or `positive`.

| Model | Approach |
|---|---|
| `textblob` | Original lexicon-based baseline using TextBlob polarity |
| `vader` | Lexicon-based sentiment model designed for social-media text |
| `pytorch` | Custom classifier trained from scratch on TweetEval |
| `finetuned` | `distilroberta-base` fine-tuned on TweetEval train (2 epochs, free Colab T4) with `scripts/finetune_transformer.py` |
| `transformer` | Pretrained `cardiffnlp/twitter-roberta-base-sentiment-latest` model used for inference |

The custom PyTorch classifier uses:

```text
Token IDs → Embedding → Masked mean pooling → Linear → ReLU → Linear → 3 classes
```

It is intentionally lightweight: the goal is a trainable neural baseline and a clear PyTorch training/evaluation workflow rather than an advanced architecture.

## Project Evolution

### Original 2024 project

The original course project used PRAW to stream Reddit submissions from `r/colors`, analyzed submission titles with TextBlob, mapped sentiment polarity to one of seven colors, and replied with a color suggestion. The original implementation is preserved in `legacy/original_2024_bot.py`.

### 2026 extension

The extended project adds:
- a common interface for interchangeable sentiment models
- VADER and pretrained Transformer baselines
- a custom trainable PyTorch sentiment classifier
- a DistilRoBERTa model fine-tuned on the same training split, for a like-for-like accuracy–latency comparison
- evaluation using accuracy, precision, recall, macro F1, confusion matrices, and inference latency
- train/validation/test separation
- rate-limit-aware retry/backoff and bot-specific duplicate-reply detection
- dry-run support, environment-variable-based credentials, and automated tests

## Evaluation Methodology

The main benchmark uses the public **TweetEval sentiment task**.

| Split | Usage |
|---|---|
| `train` | Builds the PyTorch vocabulary and trains weights for both the custom PyTorch model and the fine-tuned DistilRoBERTa |
| `val` | Monitors performance during training |
| `test` | Held out until final evaluation of all models |

The PyTorch vocabulary is built only from the training split. Validation examples are not used to update model weights, and the test split is reserved for final evaluation. Both trained models see exactly the same 45,615 training examples.

`data/smoke_test_sample.csv` is a small synthetic dataset used only to verify that the pipeline runs end to end; it is not used for reported benchmark results.

## Architecture

```text
                         ┌─ TextBlob
                         ├─ VADER
Input text ──► Model ────┼─ Custom PyTorch
                         ├─ Fine-tuned DistilRoBERTa
                         └─ Twitter-RoBERTa (reference)
              │
              ▼
     SentimentPrediction
       (label, polarity)
          │         │
          ▼         ▼
   Evaluation    Color mapping
     pipeline         │
                      ▼
                  Reddit bot
```

The offline evaluator and Reddit bot share the same model interface, so model implementations can be compared without changing the downstream application.

## Repository Structure

```text
reddit-sentiment-evaluation/
├── .github/workflows/          # CI: runs the test suite on every push
├── checkpoints/                # trained weights (gitignored)
├── data/                       # smoke-test data; TweetEval downloads are gitignored
├── legacy/                     # original 2024 bot
├── results/                    # evaluation output JSON
├── scripts/
│   ├── prepare_tweet_eval.py   # downloads TweetEval splits to CSV
│   └── finetune_transformer.py # fine-tunes DistilRoBERTa (Colab-friendly)
├── sentiment_bot/
│   ├── models/                 # TextBlob, VADER, PyTorch, fine-tuned and pretrained Transformer wrappers
│   ├── torch_sentiment/        # dataset, model, and training loop
│   ├── evaluation/             # metrics and comparison runner
│   └── reddit/                 # Reddit client and bot logic
├── tests/
├── requirements.txt
└── requirements-transformer.txt
```

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt -r requirements-transformer.txt
```

Reddit credentials are not required for offline training or evaluation. To run the bot, copy `.env.example` to `.env` and provide your own Reddit application credentials.

## Train the PyTorch Model

```bash
python scripts/prepare_tweet_eval.py --split train --out data/tweeteval_train.csv
python scripts/prepare_tweet_eval.py --split val --out data/tweeteval_val.csv
python scripts/prepare_tweet_eval.py --split test --out data/tweeteval_test.csv

python -m sentiment_bot.torch_sentiment.train --train data/tweeteval_train.csv --val data/tweeteval_val.csv --out checkpoints/pytorch_sentiment.pt
```

The training pipeline builds its vocabulary from the training split, uses `CrossEntropyLoss` and Adam optimization, reports train/validation loss and accuracy after each epoch, and saves the model weights, vocabulary, and architecture configuration.

## Fine-tune DistilRoBERTa

Fine-tuning needs a GPU; it was run on a free Colab T4 (about 15–25 minutes for 2 epochs). It uses the same train/val CSVs as the PyTorch model and never touches the test split.

```bash
pip install datasets accelerate
python scripts/finetune_transformer.py \
    --train data/tweeteval_train.csv --val data/tweeteval_val.csv \
    --out checkpoints/distilroberta-tweeteval
```

The evaluator loads the `finetuned` model from `checkpoints/distilroberta-tweeteval`. If you train on Colab, download that folder into `checkpoints/` before running the evaluation.

## Run the Evaluation

```bash
# all registered models
python -m sentiment_bot.evaluation.runner --data data/tweeteval_test.csv

# or a subset
python -m sentiment_bot.evaluation.runner --data data/tweeteval_test.csv --models pytorch finetuned
```

The evaluator reports accuracy, macro precision/recall/F1, confusion matrices, and average inference latency.

## Run the Reddit Bot

Dry run (default):

```bash
python -m sentiment_bot.reddit.bot
```

Live replies:

```bash
python -m sentiment_bot.reddit.bot --live
```

Credentials are loaded from environment variables rather than committed source files.

## Testing

```bash
pytest
```

## Limitations

- **Not a fully controlled comparison.** TextBlob and VADER need no supervised training; the custom PyTorch model and the fine-tuned DistilRoBERTa are trained on the same TweetEval training split, which makes those two a like-for-like pair. The off-the-shelf Twitter-RoBERTa was built for short social-media sentiment and is likely trained on TweetEval-style data, so it is a reference ceiling rather than a fair comparison.
- **Simple architecture.** The custom classifier uses mean pooling, so it loses word order. A Transformer uses attention, so each word can use the context of the other words.
- **Fixed thresholds.** Polarity above 0.05 is positive, below -0.05 is negative, and in between is neutral. This is VADER's standard convention; TextBlob reuses it for consistency. The cutoff was not tuned.
- **No checkpoint selection for the PyTorch model.** It trains for 5 epochs and saves the final epoch, not the best validation epoch.
- **Machine-specific latency.** Measured on a local CPU with batch size 1.
- The Reddit bot is a portfolio-scale application rather than a production-scale service.

## Future Work

- Sweep the polarity cutoff on the validation set (not the test set), focusing on the neutral class.
- Save the best-validation checkpoint or add early stopping.
- Replace mean pooling with an attention-based encoder.
- Repeat training with several random seeds and report the variance of the scores.
- Measure batched and GPU latency alongside the batch-size-1 CPU numbers.

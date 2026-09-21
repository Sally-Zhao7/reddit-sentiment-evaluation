# Reddit Sentiment Evaluation

A modular sentiment-analysis project comparing lexicon-based methods, a custom PyTorch classifier, and a pretrained Transformer on the same held-out dataset, with a Reddit bot as the original application.

This began as a 2024 course project: a Reddit bot that used TextBlob sentiment polarity to recommend colors. In May 2026, I extended it into a reproducible model-evaluation pipeline with a trainable PyTorch baseline, additional sentiment models, testing, and more reliable Reddit API handling.

## Results

All four approaches were evaluated on the same **12,284-example held-out TweetEval sentiment test split**.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Avg. Latency |
|---|---:|---:|---:|---:|---:|
| TextBlob | 0.487 | 0.495 | 0.490 | 0.464 | 0.11 ms |
| VADER | 0.530 | 0.546 | 0.570 | 0.529 | 0.04 ms |
| Custom PyTorch | 0.576 | 0.565 | 0.578 | 0.571 | 0.10 ms |
| Transformer | 0.723 | 0.720 | 0.734 | 0.725 | 17.48 ms |

The custom PyTorch classifier improved macro F1 over both lexicon-based baselines while maintaining low CPU inference latency. The pretrained Transformer achieved the strongest predictive performance, but with substantially higher latency.

Latency was measured per example on the local CPU used for this evaluation, with batch size 1, so the values should be treated as machine-specific rather than universal benchmarks.

## Models

All four models implement the same `SentimentModel` interface and output `negative`, `neutral`, or `positive`.

| Model | Approach |
|---|---|
| `textblob` | Original lexicon-based baseline using TextBlob polarity |
| `vader` | Lexicon-based sentiment model designed for social-media text |
| `pytorch` | Custom classifier trained from scratch on TweetEval |
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
- evaluation using accuracy, precision, recall, macro F1, confusion matrices, and inference latency
- train/validation/test separation
- rate-limit-aware retry/backoff and bot-specific duplicate-reply detection
- dry-run support, environment-variable-based credentials, and automated tests

## Evaluation Methodology

The main benchmark uses the public **TweetEval sentiment task**.

| Split | Usage |
|---|---|
| `train` | Builds the PyTorch vocabulary and trains model weights |
| `val` | Monitors performance during training |
| `test` | Held out until final evaluation of all four approaches |

The PyTorch vocabulary is built only from the training split. Validation examples are not used to update model weights, and the test split is reserved for final evaluation.

`data/smoke_test_sample.csv` is a small synthetic dataset used only to verify that the pipeline runs end to end; it is not used for reported benchmark results.

## Architecture

```text
                         ┌─ TextBlob
                         ├─ VADER
Input text ──► Model ────┼─ Custom PyTorch
                         └─ Transformer
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
├── data/                       # smoke-test data; TweetEval downloads are gitignored
├── legacy/                     # original 2024 bot
├── scripts/
│   └── prepare_tweet_eval.py
├── sentiment_bot/
│   ├── models/                 # TextBlob, VADER, PyTorch, Transformer wrappers
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

## Run the Evaluation

```bash
python -m sentiment_bot.evaluation.runner --data data/tweeteval_test.csv
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

This is not a fully controlled architecture comparison. TextBlob and VADER require no supervised training, the custom PyTorch model is trained on TweetEval's training distribution, and the pretrained Transformer was developed for similar short social-media sentiment data.

The custom PyTorch classifier is intentionally simple and uses mean pooling rather than attention or recurrence. TextBlob/VADER class thresholds are modeling choices, and latency measurements depend on the local CPU and single-example inference setup.

The Reddit bot is a portfolio-scale application rather than a production-scale service.

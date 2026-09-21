# Evaluation data

The evaluator (`sentiment_bot/evaluation/runner.py`) expects a CSV with
exactly two columns:

```
text,label
"I love this!",positive
"This is fine I guess.",neutral
"Worst experience ever.",negative
```

`label` must be one of `negative`, `neutral`, `positive` (lowercase).

## `smoke_test_sample.csv`

A small synthetic dataset (12 short, clearly-polarized sentences) generated
during this refactor, used only to verify that the evaluation pipeline
(loading, running each model, computing metrics, writing a comparison
table) runs end to end. **It is not used as a benchmark or for reported
model performance** - 12 examples is nowhere near enough to draw any
conclusion about which model is "better," and the sentences were written to
be unambiguous on purpose, so they don't test how a model handles
real-world ambiguity. It exists purely as a fast, offline, always-available
pipeline correctness check.

## Getting a real evaluation set

Two options, both avoiding the thing the original plan for this project
explicitly ruled out (asking an LLM to fabricate a few hundred "labeled"
examples and calling it a human-labeled benchmark):

1. **Public dataset (recommended):** run `scripts/prepare_tweet_eval.py`,
   which downloads the sentiment split of
   [TweetEval](https://github.com/cardiffnlp/tweeteval) (short, informal,
   human-annotated social-media text - a reasonable proxy for Reddit
   titles) and converts it to the CSV schema above. This needs to be run
   from a machine with normal internet access; it could not be run inside
   the sandboxed environment this project was built in (see the top-level
   README's "What I could not verify here" section).
2. **Your own small real set:** collect ~30-50 actual Reddit titles (e.g.
   from `r/colors` or wherever you plan to run the bot) and label them
   yourself. Slower, but arguably more representative of the bot's real
   use case than a Twitter dataset.

If you use an LLM to help draft candidate examples for either option, treat
them as a *starting point you review and correct yourself*, and say so in
the README's results section - don't present AI-drafted labels as a
human-labeled benchmark.

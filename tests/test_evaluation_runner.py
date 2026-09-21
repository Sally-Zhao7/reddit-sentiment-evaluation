"""Exercises the evaluation runner's CSV loading / scoring / file-writing
mechanics using a fake in-test model, so this doesn't depend on textblob,
vaderSentiment or transformers being installed. Correctness of the *real*
models is covered by tests/test_models.py (skipped there when a given
library isn't available) and by actually running
`python -m sentiment_bot.evaluation.runner` per the README.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from sentiment_bot.evaluation.runner import evaluate_model, load_labeled_csv, run
from sentiment_bot.models.base import SentimentModel, SentimentPrediction


class _AlwaysPositiveModel(SentimentModel):
    name = "always_positive"

    def predict(self, text: str) -> SentimentPrediction:
        return SentimentPrediction(label="positive", polarity=0.9)


class TestLoadLabeledCsv(unittest.TestCase):
    def test_loads_valid_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("text,label\nhello,positive\nbye,negative\n")
            rows = load_labeled_csv(path)
            self.assertEqual(rows, [("hello", "positive"), ("bye", "negative")])

    def test_rejects_bad_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("foo,bar\n1,2\n")
            with self.assertRaises(ValueError):
                load_labeled_csv(path)

    def test_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("text,label\nhello,happy\n")
            with self.assertRaises(ValueError):
                load_labeled_csv(path)


class TestEvaluateModel(unittest.TestCase):
    def test_scores_against_ground_truth(self):
        model = _AlwaysPositiveModel()
        rows = [("a", "positive"), ("b", "negative")]
        metrics, examples = evaluate_model(model, rows)
        self.assertEqual(metrics.n_examples, 2)
        self.assertEqual(metrics.accuracy, 0.5)  # got "a" right, "b" wrong
        self.assertEqual(len(examples), 2)
        self.assertTrue(examples[0]["correct"])
        self.assertFalse(examples[1]["correct"])


class TestRunEndToEnd(unittest.TestCase):
    def test_run_writes_result_json_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.csv"
            with open(data_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label"])
                writer.writerow(["good", "positive"])
                writer.writerow(["bad", "negative"])

            output_dir = tmp_path / "results"

            # Monkeypatch the registry just for this test so `run()` (which
            # looks models up by name) can find our fake model without
            # needing textblob/vader/transformers installed.
            import sentiment_bot.evaluation.runner as runner_module

            original_registry = dict(runner_module.MODEL_REGISTRY)
            runner_module.MODEL_REGISTRY["always_positive"] = _AlwaysPositiveModel
            try:
                exit_code = run(data_path, ["always_positive"], output_dir)
            finally:
                runner_module.MODEL_REGISTRY.clear()
                runner_module.MODEL_REGISTRY.update(original_registry)

            self.assertEqual(exit_code, 0)
            result_files = list(output_dir.glob("comparison_*.json"))
            self.assertEqual(len(result_files), 1)

            payload = json.loads(result_files[0].read_text())
            self.assertEqual(payload["run_metadata"]["n_examples"], 2)
            self.assertEqual(payload["metrics"][0]["model_name"], "always_positive")
            self.assertEqual(payload["metrics"][0]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

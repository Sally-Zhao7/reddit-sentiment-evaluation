import unittest

from sentiment_bot.evaluation.metrics import compute_metrics, format_comparison_table


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = ["positive", "negative", "neutral", "positive"]
        y_pred = ["positive", "negative", "neutral", "positive"]
        metrics = compute_metrics("dummy", y_true, y_pred, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.macro_f1, 1.0)
        self.assertAlmostEqual(metrics.avg_latency_ms, 2.5)
        self.assertEqual(metrics.n_examples, 4)

    def test_all_wrong_predictions(self):
        y_true = ["positive", "positive"]
        y_pred = ["negative", "negative"]
        metrics = compute_metrics("dummy", y_true, y_pred, [1.0, 1.0])
        self.assertEqual(metrics.accuracy, 0.0)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compute_metrics("dummy", ["positive"], ["positive", "negative"], [1.0, 1.0])

    def test_rejects_empty_dataset(self):
        with self.assertRaises(ValueError):
            compute_metrics("dummy", [], [], [])

    def test_confusion_matrix_shape(self):
        y_true = ["positive", "negative", "neutral"]
        y_pred = ["positive", "negative", "neutral"]
        metrics = compute_metrics("dummy", y_true, y_pred, [1.0, 1.0, 1.0])
        self.assertEqual(len(metrics.confusion_matrix), 3)
        self.assertEqual(len(metrics.confusion_matrix[0]), 3)


class TestFormatComparisonTable(unittest.TestCase):
    def test_produces_one_row_per_model(self):
        y_true = ["positive", "negative"]
        y_pred = ["positive", "negative"]
        m1 = compute_metrics("model_a", y_true, y_pred, [1.0, 1.0])
        m2 = compute_metrics("model_b", y_true, y_pred, [2.0, 2.0])
        table = format_comparison_table([m1, m2])
        self.assertIn("model_a", table)
        self.assertIn("model_b", table)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest

from sentiment_bot.models.base import SentimentPrediction, polarity_to_label


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class TestSentimentPrediction(unittest.TestCase):
    def test_rejects_invalid_label(self):
        with self.assertRaises(ValueError):
            SentimentPrediction(label="happy", polarity=0.0)

    def test_rejects_out_of_range_polarity(self):
        with self.assertRaises(ValueError):
            SentimentPrediction(label="positive", polarity=1.5)

    def test_accepts_valid_prediction(self):
        pred = SentimentPrediction(label="neutral", polarity=0.0)
        self.assertEqual(pred.label, "neutral")


class TestPolarityToLabel(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(polarity_to_label(0.5), "positive")
        self.assertEqual(polarity_to_label(-0.5), "negative")
        self.assertEqual(polarity_to_label(0.0), "neutral")
        self.assertEqual(polarity_to_label(0.05), "neutral")  # boundary is exclusive
        self.assertEqual(polarity_to_label(0.0500001), "positive")


@unittest.skipUnless(_has_module("textblob"), "textblob is not installed in this environment")
class TestTextBlobModel(unittest.TestCase):
    def setUp(self):
        from sentiment_bot.models.textblob_model import TextBlobModel

        self.model = TextBlobModel()

    def test_empty_input_is_neutral(self):
        pred = self.model.predict("")
        self.assertEqual(pred.label, "neutral")
        self.assertEqual(pred.polarity, 0.0)

    def test_clearly_positive_text(self):
        pred = self.model.predict("This is wonderful, amazing, fantastic news!")
        self.assertEqual(pred.label, "positive")


@unittest.skipUnless(_has_module("vaderSentiment"), "vaderSentiment is not installed in this environment")
class TestVaderModel(unittest.TestCase):
    def setUp(self):
        from sentiment_bot.models.vader_model import VaderModel

        self.model = VaderModel()

    def test_empty_input_is_neutral(self):
        pred = self.model.predict("   ")
        self.assertEqual(pred.label, "neutral")

    def test_clearly_negative_text(self):
        pred = self.model.predict("This is terrible, awful, disgusting.")
        self.assertEqual(pred.label, "negative")


@unittest.skipUnless(
    _has_module("transformers") and _has_module("torch"),
    "transformers/torch are not installed in this environment",
)
class TestTransformerModel(unittest.TestCase):
    def test_empty_input_is_neutral_without_loading_model(self):
        # Only test the cheap empty-input short-circuit here; loading the
        # actual pretrained model is exercised in the evaluation run, not
        # in unit tests, to keep `pytest` fast and offline-friendly.
        from sentiment_bot.models.transformer_model import TransformerModel

        model = TransformerModel.__new__(TransformerModel)  # skip __init__ (no model download)
        pred = model.predict("")
        self.assertEqual(pred.label, "neutral")


if __name__ == "__main__":
    unittest.main()

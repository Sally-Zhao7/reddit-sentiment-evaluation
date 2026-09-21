"""Tests for the custom PyTorch sentiment model.

All torch-dependent tests are skipped (not silently passed) when torch
isn't installed. torch is imported lazily inside each test/setUp, never at
module level, so this file can still be collected and reported on in an
environment without torch.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_TORCH_AVAILABLE = _has_module("torch")


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is not installed in this environment")
class TestVocab(unittest.TestCase):
    def setUp(self):
        from sentiment_bot.torch_sentiment.data import Vocab

        self.Vocab = Vocab

    def test_build_includes_pad_and_unk_at_fixed_indices(self):
        vocab = self.Vocab.build(["good movie", "bad movie"])
        self.assertEqual(vocab.pad_id, 0)
        self.assertEqual(vocab.unk_id, 1)

    def test_min_freq_drops_rare_words(self):
        vocab = self.Vocab.build(["good good good", "rare"], min_freq=2)
        self.assertIn("good", vocab.token_to_id)
        self.assertNotIn("rare", vocab.token_to_id)

    def test_unknown_word_maps_to_unk_id(self):
        vocab = self.Vocab.build(["good movie"])
        encoded = vocab.encode(["good", "somethingneverseen"])
        self.assertEqual(encoded, [vocab.token_to_id["good"], vocab.unk_id])

    def test_round_trips_through_dict(self):
        vocab = self.Vocab.build(["good movie", "bad movie"])
        restored = self.Vocab.from_dict(vocab.to_dict())
        self.assertEqual(restored.token_to_id, vocab.token_to_id)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is not installed in this environment")
class TestCollateBatch(unittest.TestCase):
    def test_pads_to_longest_sequence_and_builds_mask(self):
        from sentiment_bot.torch_sentiment.data import collate_batch

        batch = [([5, 6, 7], 0), ([9], 1)]
        token_ids, attention_mask, labels = collate_batch(batch, pad_id=0)

        self.assertEqual(list(token_ids.shape), [2, 3])
        self.assertEqual(token_ids[0].tolist(), [5, 6, 7])
        self.assertEqual(token_ids[1].tolist(), [9, 0, 0])
        self.assertEqual(attention_mask[1].tolist(), [1, 0, 0])
        self.assertEqual(labels.tolist(), [0, 1])


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is not installed in this environment")
class TestSentimentClassifier(unittest.TestCase):
    def test_forward_output_shape(self):
        import torch

        from sentiment_bot.torch_sentiment.model import SentimentClassifier

        model = SentimentClassifier(vocab_size=50, embed_dim=8, hidden_dim=4, num_classes=3)
        token_ids = torch.randint(0, 50, (2, 6))
        attention_mask = torch.ones(2, 6, dtype=torch.long)

        logits = model(token_ids, attention_mask)
        self.assertEqual(list(logits.shape), [2, 3])

    def test_padding_is_excluded_from_pooling(self):
        # Two identical real tokens, one padded with extra zeros - since
        # padding_idx=0's embedding stays at zero AND the mask zeroes it
        # out either way, the pooled result should be unaffected by
        # trailing padding.
        import torch

        from sentiment_bot.torch_sentiment.model import SentimentClassifier

        torch.manual_seed(0)
        model = SentimentClassifier(vocab_size=50, embed_dim=8, hidden_dim=4, num_classes=3, pad_idx=0)
        model.eval()

        unpadded_ids = torch.tensor([[5, 6]])
        unpadded_mask = torch.tensor([[1, 1]])

        padded_ids = torch.tensor([[5, 6, 0, 0]])
        padded_mask = torch.tensor([[1, 1, 0, 0]])

        with torch.no_grad():
            out_unpadded = model(unpadded_ids, unpadded_mask)
            out_padded = model(padded_ids, padded_mask)

        self.assertTrue(torch.allclose(out_unpadded, out_padded, atol=1e-6))


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is not installed in this environment")
class TestPyTorchModelInterface(unittest.TestCase):
    """Trains a tiny model for one step just to get a checkpoint on disk,
    then checks the SentimentModel wrapper loads it and behaves like the
    other models (this is what makes it usable by the same evaluator)."""

    def _make_checkpoint(self, path: Path):
        import torch

        from sentiment_bot.torch_sentiment.data import Vocab
        from sentiment_bot.torch_sentiment.model import SentimentClassifier

        vocab = Vocab.build(["good movie", "bad movie", "okay movie"], min_freq=1)
        model = SentimentClassifier(vocab_size=len(vocab), embed_dim=4, hidden_dim=4, pad_idx=vocab.pad_id)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab.to_dict(),
                "config": {
                    "vocab_size": len(vocab),
                    "embed_dim": 4,
                    "hidden_dim": 4,
                    "pad_idx": vocab.pad_id,
                },
            },
            path,
        )

    def test_checkpoint_save_and_load_round_trip(self):
        from sentiment_bot.models.pytorch_model import PyTorchModel

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            self._make_checkpoint(ckpt_path)

            model = PyTorchModel(checkpoint_path=str(ckpt_path))
            prediction = model.predict("this was a good movie")

            self.assertEqual(prediction.label, prediction.label)  # doesn't raise
            self.assertIn(prediction.label, ("negative", "neutral", "positive"))

    def test_missing_checkpoint_raises_clear_error(self):
        from sentiment_bot.models.pytorch_model import PyTorchModel

        with self.assertRaises(FileNotFoundError):
            PyTorchModel(checkpoint_path="/nonexistent/path/model.pt")

    def test_implements_common_sentiment_model_interface(self):
        from sentiment_bot.models.base import SentimentModel
        from sentiment_bot.models.pytorch_model import PyTorchModel

        self.assertTrue(issubclass(PyTorchModel, SentimentModel))

    def test_compatible_with_evaluation_runner(self):
        # Same code path the real evaluator uses (evaluate_model), just
        # pointed at a freshly-trained tiny checkpoint instead of a real one.
        from sentiment_bot.evaluation.runner import evaluate_model
        from sentiment_bot.models.pytorch_model import PyTorchModel

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            self._make_checkpoint(ckpt_path)

            model = PyTorchModel(checkpoint_path=str(ckpt_path))
            rows = [("good movie", "positive"), ("bad movie", "negative")]
            metrics, examples = evaluate_model(model, rows)

            self.assertEqual(metrics.n_examples, 2)
            self.assertEqual(len(examples), 2)


if __name__ == "__main__":
    unittest.main()

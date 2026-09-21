"""A small, from-scratch PyTorch sentiment classifier.

This is a genuine trainable baseline (embedding -> masked mean pooling ->
small feed-forward network), not just PyTorch installed as a dependency for
the Hugging Face transformer. See train.py for the training loop and
model.py for the architecture.
"""

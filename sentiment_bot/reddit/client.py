"""Thin wrapper around `praw.Reddit` construction.

Kept separate from bot.py so bot.py (and its tests) never need to import
praw directly - only this module does, and only inside the function body,
so importing `sentiment_bot.reddit` doesn't require praw to be installed.
"""
from __future__ import annotations

from ..config import RedditCredentials


def build_reddit_client(credentials: RedditCredentials):
    """Construct a `praw.Reddit` instance from credentials.

    Raises RuntimeError with a clear message if credentials are incomplete,
    rather than letting praw fail with a less obvious error later.
    """
    if not credentials.is_complete():
        missing = ", ".join(credentials.missing_fields())
        raise RuntimeError(
            f"Missing Reddit credentials: {missing}. "
            "Set them as environment variables (see .env.example)."
        )

    try:
        import praw
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "The Reddit bot requires the 'praw' package. Install it with: pip install praw"
        ) from exc

    return praw.Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        username=credentials.username,
        password=credentials.password,
        user_agent=credentials.user_agent,
    )

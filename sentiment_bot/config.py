"""Configuration loading.

All Reddit credentials and runtime knobs come from environment variables
(optionally via a local .env file), never from source code. This replaces
the original notebook's `%run reddit_keys.py`, which loaded plaintext
credentials from a gitignored script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    # python-dotenv is a small, optional convenience: if present and a
    # .env file exists, load it into os.environ. If it isn't installed,
    # we fall back to whatever is already in the environment.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - exercised only when dotenv is absent
    pass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class RedditCredentials:
    client_id: Optional[str] = field(default_factory=lambda: os.environ.get("REDDIT_CLIENT_ID"))
    client_secret: Optional[str] = field(default_factory=lambda: os.environ.get("REDDIT_CLIENT_SECRET"))
    username: Optional[str] = field(default_factory=lambda: os.environ.get("REDDIT_USERNAME"))
    password: Optional[str] = field(default_factory=lambda: os.environ.get("REDDIT_PASSWORD"))
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "REDDIT_USER_AGENT", "reddit-sentiment-eval bot (see README)"
        )
    )

    def is_complete(self) -> bool:
        return all([self.client_id, self.client_secret, self.username, self.password])

    def missing_fields(self) -> list[str]:
        names = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": self.password,
        }
        return [name for name, value in names.items() if not value]


@dataclass(frozen=True)
class BotConfig:
    subreddit: str = field(default_factory=lambda: os.environ.get("REDDIT_SUBREDDIT", "colors"))
    sentiment_model: str = field(default_factory=lambda: os.environ.get("SENTIMENT_MODEL", "textblob"))
    dry_run: bool = field(default_factory=lambda: _get_bool("BOT_DRY_RUN", True))
    scan_interval_seconds: float = field(default_factory=lambda: _get_float("SCAN_INTERVAL_SECONDS", 10.0))
    max_retries: int = field(default_factory=lambda: _get_int("MAX_RETRIES", 5))
    base_retry_delay_seconds: float = field(
        default_factory=lambda: _get_float("BASE_RETRY_DELAY_SECONDS", 5.0)
    )
    max_retry_delay_seconds: float = field(
        default_factory=lambda: _get_float("MAX_RETRY_DELAY_SECONDS", 600.0)
    )
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    reply_template: str = "How about the color {color}?"


def load_credentials() -> RedditCredentials:
    return RedditCredentials()


def load_bot_config() -> BotConfig:
    return BotConfig()

"""Reddit bot loop: stream submissions, predict sentiment, suggest a color.

This is the automation half of the original 2024 notebook
(`reply_to_questions` / the `__main__` loop), refactored to:
  - use the shared sentiment-model interface instead of calling TextBlob directly
  - fix the duplicate-reply check (see reddit/replies.py)
  - retry with backoff instead of a fixed `time.sleep(60)`
  - default to a dry run (log what it *would* reply, without calling the API)

Run with `--live` to actually post replies.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from ..colors import get_color_by_mood
from ..config import BotConfig, load_bot_config, load_credentials
from ..models import SentimentModel, get_model
from ..utils import RetryConfig, configure_logging, is_retryable_error, retry_call
from .client import build_reddit_client
from .replies import already_replied_by_bot, build_reply_text, contains_keywords

logger = logging.getLogger("sentiment_bot.bot")


def process_submission(
    submission,
    *,
    model: SentimentModel,
    config: BotConfig,
    bot_username: str,
    seen_submission_ids: set[str],
    dry_run: bool,
    sleep=time.sleep,
) -> None:
    text = submission.title

    if not contains_keywords(text):
        return
    if already_replied_by_bot(submission, bot_username, seen_submission_ids):
        logger.debug("Skipping %s: already replied", submission.id)
        return

    start = time.perf_counter()
    prediction = model.predict(text)
    latency_ms = (time.perf_counter() - start) * 1000
    color = get_color_by_mood(prediction.polarity)
    reply_text = build_reply_text(color, config.reply_template)

    logger.info(
        "submission=%s model=%s label=%s color=%s latency_ms=%.1f",
        submission.id, model.name, prediction.label, color, latency_ms,
    )

    if dry_run:
        logger.info("[dry run] would reply to %s: %r", submission.id, reply_text)
        seen_submission_ids.add(submission.id)
        return

    def _do_reply():
        submission.reply(reply_text)

    def _on_retry(attempt, exc, wait_seconds):
        logger.warning("Retry %d for %s after %s (waiting %.0fs)", attempt, submission.id, exc, wait_seconds)

    try:
        retry_call(
            _do_reply,
            config=RetryConfig(
                max_retries=config.max_retries,
                base_delay_seconds=config.base_retry_delay_seconds,
                max_delay_seconds=config.max_retry_delay_seconds,
            ),
            is_retryable=is_retryable_error,
            sleep=sleep,
            on_retry=_on_retry,
        )
        seen_submission_ids.add(submission.id)
        logger.info("Replied to %s with color %s", submission.id, color)
    except Exception as exc:
        # Non-retryable (e.g. locked thread) or retries exhausted: log and
        # move on to the next submission rather than crashing the loop.
        logger.error("Giving up on %s: %s", submission.id, exc)


def run_bot(
    config: BotConfig,
    model: SentimentModel,
    reddit,
    bot_username: str,
    dry_run: bool,
    sleep=time.sleep,
) -> None:
    """Run the streaming loop. Intended to run until interrupted (Ctrl+C)."""
    seen_submission_ids: set[str] = set()
    subreddit = reddit.subreddit(config.subreddit)
    logger.info(
        "Starting bot: subreddit=%s model=%s dry_run=%s", config.subreddit, model.name, dry_run
    )

    for submission in subreddit.stream.submissions():
        try:
            process_submission(
                submission,
                model=model,
                config=config,
                bot_username=bot_username,
                seen_submission_ids=seen_submission_ids,
                dry_run=dry_run,
                sleep=sleep,
            )
        except Exception as exc:
            # Mirrors the original notebook's outer "don't let one bad
            # submission kill the whole loop" behavior, but logs the
            # specific failure instead of a blanket 60-second sleep.
            logger.error("Failed to process submission %s: %s", getattr(submission, "id", None), exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Actually post replies (default: dry run, log only)")
    parser.add_argument("--subreddit", default=None, help="Override REDDIT_SUBREDDIT")
    parser.add_argument("--model", default=None, help="Override SENTIMENT_MODEL")
    args = parser.parse_args(argv)

    config = load_bot_config()
    credentials = load_credentials()
    if args.subreddit:
        config = BotConfig(**{**config.__dict__, "subreddit": args.subreddit})
    model_name = args.model or config.sentiment_model

    configure_logging(config.log_level)

    try:
        model = get_model(model_name)
    except Exception as exc:
        print(f"Failed to load model {model_name!r}: {exc}", file=sys.stderr)
        return 1

    try:
        reddit = build_reddit_client(credentials)
    except Exception as exc:
        print(f"Failed to build Reddit client: {exc}", file=sys.stderr)
        return 1

    dry_run = config.dry_run and not args.live
    try:
        run_bot(config, model, reddit, credentials.username, dry_run=dry_run)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

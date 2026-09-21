"""Pure reply-decision logic: no praw import, no network calls.

Kept separate so it's cheap to unit test with small fake submission/comment
objects instead of a live (or even mocked) praw client. `submission` and
`comment` here are just expected to look like praw's own objects (a
`submission.id`, `submission.comments.replace_more(limit)` /
`.list()`, and each comment having `.author.name` and `.body`) - plain duck
typing, no formal interface needed for objects this small.
"""
from __future__ import annotations


def contains_keywords(text: str) -> bool:
    """Preserved from the original notebook: currently a no-op filter.

    The original 2024 bot replied to every submission title regardless of
    content; this function is kept as the extension point where real
    keyword/topic filtering could be added later (e.g. only reply to
    questions), without changing the calling code in bot.py.
    """
    return True


def already_replied_by_bot(submission, bot_username: str, seen_submission_ids: set[str]) -> bool:
    """True if this bot has already replied to `submission`.

    Two checks, cheapest first:
    1. An in-memory set of submission ids handled earlier in this process
       run (avoids a comments.list() API call for posts we just replied to).
    2. The submission's own comments, restricted to ones authored by
       `bot_username`.

    This fixes a bug in the original notebook's `already_replied`, which
    matched the reply text against *any* commenter's text - so a human
    typing "How about the color blue?" as a joke would have permanently
    silenced the bot on that thread.
    """
    if submission.id in seen_submission_ids:
        return True

    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        author = getattr(comment, "author", None)
        author_name = getattr(author, "name", None)
        if author_name == bot_username and "How about the color" in comment.body:
            return True
    return False


def build_reply_text(color: str, template: str = "How about the color {color}?") -> str:
    return template.format(color=color)

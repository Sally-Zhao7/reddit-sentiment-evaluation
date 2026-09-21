import unittest

from sentiment_bot.reddit.replies import already_replied_by_bot, build_reply_text, contains_keywords


class _FakeAuthor:
    def __init__(self, name):
        self.name = name


class _FakeComment:
    def __init__(self, author_name, body):
        self.author = _FakeAuthor(author_name) if author_name else None
        self.body = body


class _FakeCommentForest:
    def __init__(self, comments):
        self._comments = comments
        self.replace_more_called_with = None

    def replace_more(self, limit):
        self.replace_more_called_with = limit

    def list(self):
        return self._comments


class _FakeSubmission:
    def __init__(self, id, comments):
        self.id = id
        self.comments = _FakeCommentForest(comments)


class TestContainsKeywords(unittest.TestCase):
    def test_always_true_preserving_original_behavior(self):
        self.assertTrue(contains_keywords("anything at all"))
        self.assertTrue(contains_keywords(""))


class TestBuildReplyText(unittest.TestCase):
    def test_default_template(self):
        self.assertEqual(build_reply_text("Blue"), "How about the color Blue?")

    def test_custom_template(self):
        self.assertEqual(build_reply_text("Red", "Try {color}!"), "Try Red!")


class TestAlreadyRepliedByBot(unittest.TestCase):
    def test_true_when_submission_id_already_seen_this_run(self):
        submission = _FakeSubmission("abc", [])
        seen = {"abc"}
        self.assertTrue(already_replied_by_bot(submission, "my_bot", seen))
        # Fast path: shouldn't even need to touch comments.
        self.assertIsNone(submission.comments.replace_more_called_with)

    def test_true_when_bot_already_commented(self):
        comments = [_FakeComment("my_bot", "How about the color Blue?")]
        submission = _FakeSubmission("abc", comments)
        self.assertTrue(already_replied_by_bot(submission, "my_bot", set()))

    def test_false_when_only_a_human_used_similar_wording(self):
        # Regression test for the original notebook's bug: matching on text
        # content regardless of author would incorrectly treat this as
        # "already replied".
        comments = [_FakeComment("random_human", "How about the color Blue? lol")]
        submission = _FakeSubmission("abc", comments)
        self.assertFalse(already_replied_by_bot(submission, "my_bot", set()))

    def test_false_when_bot_commented_something_unrelated(self):
        comments = [_FakeComment("my_bot", "unrelated comment")]
        submission = _FakeSubmission("abc", comments)
        self.assertFalse(already_replied_by_bot(submission, "my_bot", set()))

    def test_false_when_no_comments(self):
        submission = _FakeSubmission("abc", [])
        self.assertFalse(already_replied_by_bot(submission, "my_bot", set()))

    def test_calls_replace_more_with_limit_zero(self):
        submission = _FakeSubmission("abc", [])
        already_replied_by_bot(submission, "my_bot", set())
        self.assertEqual(submission.comments.replace_more_called_with, 0)


if __name__ == "__main__":
    unittest.main()

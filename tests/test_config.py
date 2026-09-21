import os
import unittest

from sentiment_bot.config import BotConfig, RedditCredentials


class TestRedditCredentials(unittest.TestCase):
    def test_incomplete_when_missing_fields(self):
        creds = RedditCredentials(client_id=None, client_secret="x", username="y", password=None)
        self.assertFalse(creds.is_complete())
        self.assertEqual(set(creds.missing_fields()), {"client_id", "password"})

    def test_complete_when_all_present(self):
        creds = RedditCredentials(client_id="a", client_secret="b", username="c", password="d")
        self.assertTrue(creds.is_complete())
        self.assertEqual(creds.missing_fields(), [])


class TestBotConfigDefaults(unittest.TestCase):
    def test_defaults_are_safe(self):
        # dry_run must default to True so a fresh checkout never posts
        # to Reddit until someone explicitly opts in.
        env_backup = {k: os.environ.pop(k, None) for k in ["BOT_DRY_RUN", "REDDIT_SUBREDDIT"]}
        try:
            config = BotConfig()
            self.assertTrue(config.dry_run)
            self.assertEqual(config.subreddit, "colors")
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()

import unittest

from sentiment_bot.utils import (
    RetryConfig,
    extract_wait_seconds,
    is_retryable_error,
    retry_call,
)


class TestExtractWaitSeconds(unittest.TestCase):
    def test_parses_minutes_from_real_reddit_message(self):
        # Verbatim message from the original notebook's captured run log.
        msg = 'RATELIMIT: "Looks like you\'ve been doing that a lot. Take a break for 9 minutes before trying again." on field \'ratelimit\''
        self.assertEqual(extract_wait_seconds(msg), 540.0)

    def test_parses_seconds(self):
        self.assertEqual(extract_wait_seconds("Take a break for 30 seconds before trying again."), 30.0)

    def test_returns_none_for_unrelated_message(self):
        self.assertIsNone(extract_wait_seconds("some unrelated error"))


class TestIsRetryableError(unittest.TestCase):
    def test_rate_limit_is_retryable(self):
        self.assertTrue(is_retryable_error(Exception("RATELIMIT: slow down")))

    def test_timeout_is_retryable(self):
        self.assertTrue(is_retryable_error(Exception("Connection timeout")))

    def test_value_error_is_not_retryable(self):
        self.assertFalse(is_retryable_error(ValueError("thread is locked")))


class TestRetryCall(unittest.TestCase):
    def test_succeeds_without_retry(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(retry_call(fn, sleep=lambda s: None), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_transient_error_using_reddits_own_wait_time(self):
        attempts = {"count": 0}
        sleeps = []

        def fn():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise Exception("RATELIMIT: Take a break for 1 seconds before trying again.")
            return "ok"

        result = retry_call(fn, sleep=sleeps.append)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(sleeps, [1.0, 1.0])  # parsed from the message both times

    def test_does_not_retry_non_transient_error(self):
        attempts = {"count": 0}

        def fn():
            attempts["count"] += 1
            raise ValueError("thread is locked")

        with self.assertRaises(ValueError):
            retry_call(fn, sleep=lambda s: None)
        self.assertEqual(attempts["count"], 1)

    def test_gives_up_after_max_retries(self):
        attempts = {"count": 0}

        def fn():
            attempts["count"] += 1
            raise Exception("RATELIMIT")

        with self.assertRaises(Exception):
            retry_call(fn, config=RetryConfig(max_retries=2, base_delay_seconds=0), sleep=lambda s: None)
        self.assertEqual(attempts["count"], 3)  # initial try + 2 retries


if __name__ == "__main__":
    unittest.main()

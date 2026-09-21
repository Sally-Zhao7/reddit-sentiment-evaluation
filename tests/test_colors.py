import unittest

from sentiment_bot.colors import get_color_by_mood


class TestColorMapping(unittest.TestCase):
    def test_boundaries_match_original_thresholds(self):
        cases = [
            (-1.0, "Green"),
            (-0.8, "Green"),
            (-0.79, "Teal"),
            (-0.4, "Teal"),
            (-0.39, "Blue"),
            (0.0, "Blue"),
            (0.01, "Yellow"),
            (0.4, "Yellow"),
            (0.41, "Orange"),
            (0.7, "Orange"),
            (0.71, "Red"),
            (0.9, "Red"),
            (0.91, "Purple"),
            (1.0, "Purple"),
        ]
        for polarity, expected in cases:
            with self.subTest(polarity=polarity):
                self.assertEqual(get_color_by_mood(polarity), expected)


if __name__ == "__main__":
    unittest.main()

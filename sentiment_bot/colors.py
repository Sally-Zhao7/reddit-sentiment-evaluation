"""Polarity -> color mapping.

This is the original 2024 project's core idea, preserved exactly: a
continuous sentiment polarity score in [-1, 1] is bucketed into one of
seven colors. The thresholds and color names are unchanged from the
original notebook.
"""
from __future__ import annotations

from enum import Enum


class Color(Enum):
    GREEN = "Green"
    TEAL = "Teal"
    BLUE = "Blue"
    YELLOW = "Yellow"
    ORANGE = "Orange"
    RED = "Red"
    PURPLE = "Purple"


def get_color_by_mood(mood_value: float) -> str:
    """Return a color name for a mood value ranging from -1 (calm) to 1 (crazy).

    Identical thresholds to the original 2024 notebook.
    """
    if mood_value <= -0.8:
        return Color.GREEN.value
    elif mood_value <= -0.4:
        return Color.TEAL.value
    elif mood_value <= 0:
        return Color.BLUE.value
    elif mood_value <= 0.4:
        return Color.YELLOW.value
    elif mood_value <= 0.7:
        return Color.ORANGE.value
    elif mood_value <= 0.9:
        return Color.RED.value
    else:
        return Color.PURPLE.value

"""Display units for each metric statistic, as Stage 1 writes them."""

from __future__ import annotations

UNIT_OF = {
    "sleep_efficiency_pct": "%",
    "sleep_onset_clock_min": "clock_minute",
    "final_awakening_clock_min": "clock_minute",
    "final_wake_clock_min": "clock_minute",
    "sleep_midpoint_clock_min": "clock_minute",
    "wake_episode_over_5_min_count": "count/night",
}


def unit_for(metric_id: str, statistic: str) -> str:
    """A difference keeps its metric's dimension, except clock times.

    `sleep_onset_clock_min.weekpart.weekend_minus_weekday = 94` is "1 hour 34 minutes later",
    and rendering it as a clock gives "1:34 am later" -- so a clock difference is a duration.
    A count difference is still a count and an efficiency difference is still percentage
    points; rendering either as minutes states a fact that is not in the record. The build
    side reads the unit from Stage 1's own annotation and gets all of this right; only this
    inference-side table has to reproduce it. (`final_awakening_clock_min` is accepted as an alias of `final_wake_clock_min`.)
    """
    base = UNIT_OF.get(metric_id, "min")
    if statistic == "n":
        return "night"
    if "minus" in statistic:
        return "min" if base == "clock_minute" else base
    return base

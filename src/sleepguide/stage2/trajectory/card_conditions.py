"""Card applicability from P: reference thresholds and age ranges.

Eight cards carry `reference_values` (metric, comparator, threshold) and six are age-specific;
both are facts about the packet. Cards gated on a `required_condition` the packet cannot
observe (for example caffeine use) are treated as always admissible.
"""

from __future__ import annotations

import json
from functools import lru_cache

from sleepguide.paths import EVIDENCE_BANK

# The age ranges are stated in each card's prose rather than as a field, so they are declared
# here and checked against the profile. Verified against the bank's own `select_when` text.
AGE_RANGE = {
    "adults-18-to-64-usually-need-at-least-7-hours": (18, 64),
    "adults-65-plus-usually-need-at-least-7-hours": (65, None),
    "waso-quality-reference-for-adults-18-to-64": (18, 64),
    "waso-quality-reference-for-adults-65-plus": (65, None),
    "long-wake-episode-reference-for-adults-18-to-64": (18, 64),
    "long-wake-episode-reference-for-adults-65-plus": (65, None),
}


@lru_cache(maxsize=1)
def bank() -> dict[str, dict]:
    """The 22 runtime evidence cards (identical in every Stage-1 run)."""
    entries = json.loads((EVIDENCE_BANK / "generation_bank.json").read_text(encoding="utf-8"))
    return {entry["evidence_key"]: entry for entry in entries}


def gated_cards() -> set[str]:
    """Cards whose applicability the packet can decide."""
    return {key for key, entry in bank().items() if entry.get("reference_values")}


def _recent_centre(packet: dict, metric_id: str) -> float | None:
    insight = packet.get("longitudinal_insight") or {}
    for entry in insight.get("recent_7_night_comparison") or []:
        if entry.get("metric_id") == metric_id:
            return (entry.get("recent_7_nights") or {}).get("center")
    return None


def condition_holds(packet: dict, key: str) -> bool:
    """Whether a gated card's own threshold and age range hold for this packet."""
    entry = bank().get(key)
    if entry is None or not entry.get("reference_values"):
        return True
    if key in AGE_RANGE:
        low, high = AGE_RANGE[key]
        age = (packet.get("profile") or {}).get("age_at_case_start_years")
        if age is None or age < low or (high is not None and age > high):
            return False
    for reference in entry["reference_values"]:
        value = _recent_centre(packet, reference["metric_id"])
        if value is None:
            return False
        # `at_least` states the healthy floor, so the card applies when the case is below it.
        if reference["comparator"] == "at_least" and not value < reference["value"]:
            return False
        if reference["comparator"] == "at_most" and not value > reference["value"]:
            return False
    return True


def admissible(packet: dict, key: str) -> bool:
    return True if key not in gated_cards() else condition_holds(packet, key)

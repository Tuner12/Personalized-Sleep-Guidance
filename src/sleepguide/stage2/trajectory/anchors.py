"""Resolve a numeric anchor `metric_id.origin.statistic` to its value using only P.

Anchors are the numbers Stage 1 certified against the wearable record. Resolving them from
the profile packet alone (not from Stage-1 output) is what lets the observation renderer and
the selectors run at inference time.
"""

from __future__ import annotations

import json
import re
from typing import Any


def packet_of(record_input: str) -> dict[str, Any]:
    """The JSON payload out of a rendered prompt."""
    body = record_input.split("Input payload (JSON):", 1)[-1]
    match = re.search(r"\{.*\}", body, re.S)
    if not match:
        raise ValueError("no JSON payload in the prompt")
    return json.loads(match.group(0))


def _by_metric(entries: list[dict], metric_id: str) -> dict | None:
    for entry in entries or []:
        if entry.get("metric_id") == metric_id:
            return entry
    return None


def resolve(packet: dict, metric_id: str, origin: str, statistic: str,
            day_index: int | None = None) -> float | None:
    """The packet's own value for one anchor key, or None if the packet does not carry it."""
    insight = packet.get("longitudinal_insight") or {}
    profile = packet.get("profile") or {}

    if origin == "aggregate":
        entry = _by_metric(insight.get("whole_record_summary"), metric_id)
        return None if entry is None else entry.get(statistic)

    if origin == "weekpart":
        entry = _by_metric(profile.get("weekday_weekend_comparisons"), metric_id)
        return None if entry is None else entry.get(statistic)

    if origin == "insight":
        entry = _by_metric(insight.get("recent_7_night_comparison"), metric_id)
        if entry is None:
            return None
        windows = {
            "recent_7_center": ("recent_7_nights", "center"),
            "previous_7_center": ("previous_7_nights", "center"),
            "earlier_history_center": ("earlier_history", "center"),
        }
        if statistic in windows:
            block, field = windows[statistic]
            return (entry.get(block) or {}).get(field)
        # The deltas are carried by the packet under their own names, and they must be read
        # rather than recomputed. Subtracting the two centres is wrong for clock metrics that
        # cross midnight: `sleep_onset_clock_min` with a recent centre of 58 and a previous of
        # 1408 is +90 minutes, not -1350, and the packet's stored field already wraps.
        fields = {
            "recent_minus_previous_7": "recent_minus_previous",
            "recent_minus_earlier_history": "recent_minus_earlier_history",
        }
        field = fields.get(statistic, statistic)
        return entry.get(field)

    if origin == "daily_observation":
        if day_index is None:
            return None
        for series in (insight.get("rolling_7_night_trajectory") or []):
            if series.get("metric_id") != metric_id:
                continue
            points = series.get("series") or []
            if 0 <= day_index < len(points):
                point = points[day_index]
                return point[-1] if isinstance(point, list) else point
        context = insight.get("latest_night_context") or {}
        entry = _by_metric(context.get("metrics") if isinstance(context, dict) else None,
                           metric_id)
        return None if entry is None else entry.get(statistic)

    return None

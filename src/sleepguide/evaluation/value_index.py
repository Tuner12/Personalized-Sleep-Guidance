"""Index every numeric value the profile packet supports, and extract numeric mentions from text (used by Value F1)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sleepguide.core.contracts import LongitudinalInsight, SleepProfile


def _prune(value: object) -> object:
    """Drop nulls and empty containers so the packet spends no tokens on absence."""

    if isinstance(value, dict):
        cleaned = {key: _prune(item) for key, item in value.items() if item is not None}
        return {key: item for key, item in cleaned.items() if item not in ({}, [])}
    if isinstance(value, list):
        return [_prune(item) for item in value]
    return value


def _summary(item, *, compact: bool) -> dict:
    fields = (
        ("n", "median", "mean", "minimum", "maximum")
        if compact
        else (
            "n",
            "mean",
            "median",
            "sd",
            "mad",
            "iqr",
            "p10",
            "p90",
            "minimum",
            "maximum",
            "first_value",
            "last_value",
        )
    )
    payload = {name: getattr(item, name) for name in fields}
    if not compact and item.caveats:
        payload["caveats"] = list(item.caveats)
    return payload


@dataclass(frozen=True)
class CanonicalValue:
    """One value a reader-visible number is permitted to assert."""

    metric_id: str
    origin: str
    statistic: str
    value: float
    unit: str


_AGGREGATE_FIELDS = (
    "n",
    "mean",
    "median",
    "sd",
    "mad",
    "iqr",
    "p10",
    "p90",
    "minimum",
    "maximum",
    "first_value",
    "last_value",
)
# A dispersion or difference of a clock metric is an elapsed duration, matching the unit
# normalization the Stage-1 anchor contract applies.
_CLOCK_TO_DURATION = {
    "sd",
    "mad",
    "iqr",
    # The window-scoped spreads were omitted while the whole-record ones above were not, so a
    # clock metric's `recent_window_mad` was indexed as a clock time. The value is in the
    # packet the model sees, under `longitudinal_windows`; only its unit label disagreed
    # with the anchor contract, and the export dropped the case as if the number were
    # unavailable. That cost two of the 1500 cases.
    "earlier_window_mad",
    "earlier_window_iqr",
    "middle_window_mad",
    "middle_window_iqr",
    "recent_window_mad",
    "recent_window_iqr",
    "recent_minus_earlier",
    "recent_minus_earlier_mad",
    "slope_per_7_days",
    "weekend_minus_weekday",
    "recent_minus_previous_7",
    "recent_minus_earlier_history",
    "latest_minus_prior_7",
    "latest_minus_earlier_history",
    "difference_from_full_record_center",
    "difference_from_segment_center",
    "high_minus_low_sleep",
}


def _duration_unit(unit: str, statistic: str) -> str:
    return "min" if unit == "clock_minute" and statistic in _CLOCK_TO_DURATION else unit


def canonical_value_index(
    profile: SleepProfile, insight: LongitudinalInsight
) -> list[CanonicalValue]:
    """Enumerate every value the packet exposes, with its metric and unit.

    This is the deterministic ground truth for both directions of the numeric check:
    a generated number is supported when it matches an entry, and a certified anchor is
    covered when its own value appears here.
    """

    values: list[CanonicalValue] = []

    def add(metric_id: str, origin: str, statistic: str, value, unit: str) -> None:
        if value is None:
            return
        values.append(
            CanonicalValue(
                metric_id=metric_id,
                origin=origin,
                statistic=statistic,
                value=float(value),
                unit=_duration_unit(unit, statistic),
            )
        )

    for item in profile.metric_summaries:
        for name in _AGGREGATE_FIELDS:
            unit = "night" if name == "n" else item.unit
            add(item.metric_id, "aggregate", name, getattr(item, name), unit)
    for item in profile.longitudinal_summaries:
        add(item.metric_id, "longitudinal", "n", item.n, "night")
        add(item.metric_id, "longitudinal", "span_days", item.span_days, "day")
        add(
            item.metric_id,
            "longitudinal",
            "sampling_density_pct",
            item.sampling_density_pct,
            "%",
        )
        for name in (
            "recent_minus_earlier",
            "recent_minus_earlier_mad",
            "slope_per_7_days",
        ):
            add(item.metric_id, "longitudinal", name, getattr(item, name), item.unit)
        for prefix in ("earlier_window", "middle_window", "recent_window"):
            window = getattr(item, prefix)
            if window is None:
                continue
            add(item.metric_id, "longitudinal", f"{prefix}_center", window.center, item.unit)
            add(item.metric_id, "longitudinal", f"{prefix}_mad", window.mad, item.unit)
            add(item.metric_id, "longitudinal", f"{prefix}_iqr", window.iqr, item.unit)
            add(item.metric_id, "longitudinal", f"{prefix}_n", window.n, "night")
            add(
                item.metric_id,
                "longitudinal",
                f"{prefix}_calendar_span_days",
                window.calendar_span_days,
                "day",
            )
        for event in item.episodic_deviations:
            add(item.metric_id, "longitudinal", "episodic_value", event.value, item.unit)
            add(
                item.metric_id,
                "longitudinal",
                "difference_from_segment_center",
                event.difference_from_segment_center,
                item.unit,
            )
    for item in profile.weekday_weekend_comparisons:
        if not item.reportable:
            continue
        add(item.metric_id, "weekpart", "weekday_n", item.weekday_n, "night")
        add(item.metric_id, "weekpart", "weekend_n", item.weekend_n, "night")
        for name in ("weekday_center", "weekend_center", "weekend_minus_weekday"):
            add(item.metric_id, "weekpart", name, getattr(item, name), item.unit)
    for item in profile.daily_context_associations:
        if not item.reportable:
            continue
        add(item.context_metric_id, "association", "paired_nights", item.paired_nights, "night")
        add(item.context_metric_id, "association", "spearman_rho", item.spearman_rho, "rho")
        add(
            item.context_metric_id,
            "association",
            "high_minus_low_sleep",
            item.high_minus_low_sleep,
            item.unit_sleep,
        )
    for item in insight.rolling_7_night_trajectory:
        for window in item.windows:
            add(item.metric_id, "insight", "rolling_window_center", window.center, item.unit)
    for item in insight.recent_7_night_comparison:
        for prefix, window in (
            ("recent_7", item.recent_7_nights),
            ("previous_7", item.previous_7_nights),
            ("earlier_history", item.earlier_history),
        ):
            if window is None:
                continue
            add(item.metric_id, "insight", f"{prefix}_center", window.center, item.unit)
            add(item.metric_id, "insight", f"{prefix}_n", window.n, "night")
        add(
            item.metric_id,
            "insight",
            "recent_minus_previous_7",
            item.recent_minus_previous,
            item.unit,
        )
        add(
            item.metric_id,
            "insight",
            "recent_minus_earlier_history",
            item.recent_minus_earlier_history,
            item.unit,
        )
    for item in insight.latest_night_context:
        add(item.metric_id, "insight", "latest_value", item.latest_value, item.unit)
        for prefix, window in (
            ("prior_7", item.prior_7_nights),
            ("latest_earlier_history", item.earlier_history),
        ):
            if window is None:
                continue
            add(item.metric_id, "insight", f"{prefix}_center", window.center, item.unit)
        add(
            item.metric_id,
            "insight",
            "latest_minus_prior_7",
            item.latest_minus_prior_7,
            item.unit,
        )
        add(
            item.metric_id,
            "insight",
            "latest_minus_earlier_history",
            item.latest_minus_earlier_history,
            item.unit,
        )
    for event in insight.fluctuation_points:
        add(event.metric_id, "insight", "fluctuation_value", event.value, event.unit)
        add(
            event.metric_id,
            "insight",
            "difference_from_full_record_center",
            event.difference_from_full_record_center,
            event.unit,
        )
    add("observation_window", "case", "valid_main_sleep_nights", profile.valid_main_sleep_nights, "night")
    add("observation_window", "case", "calendar_span_days", profile.calendar_span_days, "day")
    add("observation_window", "case", "coverage_pct", profile.coverage_pct, "%")
    add("case_context", "case", "age_at_case_start_years", profile.age_at_case_start_years, "year")
    return values


_NUMBER = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:,\d{3})*(?:\.\d+)?)(?![0-9])")
_HOURS_MINUTES = re.compile(
    r"\b(\d+)\s*(?:h|hr|hrs|hour|hours)\b(?:\s*(?:and\s*)?(\d+)\s*(?:m|min|mins|minute|minutes)\b)?",
    re.I,
)
_MINUTES = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", re.I)
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?", re.I)
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")


def duration_spans(text: str) -> list[tuple[int, int, float]]:
    """Return ``(start, end, minutes)`` for every elapsed-duration mention."""

    spans: list[tuple[int, int, float]] = []
    consumed: list[tuple[int, int]] = []
    for match in _HOURS_MINUTES.finditer(text):
        minutes = int(match.group(1)) * 60 + (int(match.group(2)) if match.group(2) else 0)
        spans.append((match.start(), match.end(), float(minutes)))
        consumed.append((match.start(), match.end()))
    for match in _MINUTES.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        spans.append((match.start(), match.end(), float(match.group(1))))
    return spans


def clock_spans(text: str) -> list[tuple[int, int, float]]:
    """Return ``(start, end, minute_of_day)`` for every clock time mention."""

    spans: list[tuple[int, int, float]] = []
    for match in _CLOCK.finditer(text):
        hour = int(match.group(1)) % 12
        if match.group(3).lower() == "p":
            hour += 12
        spans.append((match.start(), match.end(), float(hour * 60 + int(match.group(2)))))
    return spans


def percent_spans(text: str) -> list[tuple[int, int, float]]:
    return [(m.start(), m.end(), float(m.group(1))) for m in _PERCENT.finditer(text)]


def bare_number_spans(text: str) -> list[tuple[int, int, float]]:
    """Return plain numbers that are not part of a duration, clock time, or percentage."""

    taken: list[tuple[int, int]] = [
        (start, end)
        for start, end, _ in duration_spans(text) + clock_spans(text) + percent_spans(text)
    ]
    spans: list[tuple[int, int, float]] = []
    for match in _NUMBER.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        spans.append((match.start(), match.end(), float(match.group(1).replace(",", ""))))
    return spans


def numeric_mentions(text: str) -> list[tuple[int, int, float, str]]:
    """Return every numeric mention as ``(start, end, value, kind)``.

    ``kind`` is the surface form, which fixes the comparison unit: a duration mention can
    only be checked against a minute-valued statistic, a clock mention only against a
    clock metric.  Comparing across kinds is how a numeric audit manufactures agreement.
    """

    mentions = [(s, e, v, "duration") for s, e, v in duration_spans(text)]
    mentions += [(s, e, v, "clock") for s, e, v in clock_spans(text)]
    mentions += [(s, e, v, "percent") for s, e, v in percent_spans(text)]
    mentions += [(s, e, v, "bare") for s, e, v in bare_number_spans(text)]
    return sorted(mentions)


_KIND_UNITS = {
    "duration": {"min"},
    "clock": {"clock_minute"},
    "percent": {"%", "proportion"},
    "bare": {"night", "day", "%", "rho", "steps/day", "min/day", "beats/min", "year", "min"},
}


def supporting_values(
    value: float, kind: str, index: list[CanonicalValue], *, tolerance: float = 0.5
) -> list[CanonicalValue]:
    """Return canonical values a mention of ``value`` could faithfully be reporting.

    Duration and clock mentions are matched on the rounded minute because the reader text
    states ``6 hours 38 minutes`` for 398.5 minutes.  A percentage is matched at one
    decimal.  Anything else must agree to within ``tolerance``.
    """

    units = _KIND_UNITS[kind]
    out: list[CanonicalValue] = []
    for entry in index:
        if entry.unit not in units:
            continue
        if kind == "clock":
            if abs((round(entry.value) % 1440) - (round(value) % 1440)) <= 1:
                out.append(entry)
            continue
        if kind == "duration":
            if abs(round(abs(entry.value)) - round(value)) <= 1:
                out.append(entry)
            continue
        if abs(abs(entry.value) - value) <= tolerance:
            out.append(entry)
    return out

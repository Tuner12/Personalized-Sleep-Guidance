"""Metric definitions used by the scorer, kept as they were when the paper's numbers were computed."""

from __future__ import annotations

from dataclasses import dataclass

STAGE_CAVEAT = (
    "Sleep-stage fields are device estimates. Describe them only as recorded stage "
    "estimates; do not use them for diagnosis or as a behavior target."
)
AWAKE_CAVEAT = "Recorded awake time depends on the device's sleep/wake classification algorithm."
EFFICIENCY_CAVEAT = (
    "Sleep efficiency is transparently calculated as total sleep divided by time in bed; "
    "the source device's separate efficiency field is retained only for discrepancy checks."
)
ACTIVITY_LEVEL_CAVEAT = (
    "Activity-level minutes use vendor categories and are not a direct conversion to a "
    "clinical or public-health activity dose."
)
ACTIVITY_CALORIE_CAVEAT = (
    "Activity calories are a device estimate. Do not infer calorie intake, weight "
    "change, or an exercise dose from this field."
)
DAILY_HEART_RATE_CAVEAT = (
    "This is a daily heart-rate aggregate. It is not resting heart rate, sleeping "
    "heart rate, or heart-rate variability."
)
HEART_RATE_COVERAGE_CAVEAT = (
    "This field describes heart-rate data availability or sampling, not physiology."
)
HEART_RATE_ZONE_CAVEAT = (
    "Heart-rate-zone minutes use device-defined personalized categories. They are not "
    "clinical heart-rate abnormalities or a direct public-health activity dose."
)


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    unit: str
    is_clock: bool = False
    derived_from: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


def _metric(
    metric_id: str,
    unit: str,
    *,
    is_clock: bool = False,
    derived_from: tuple[str, ...] = (),
    caveats: tuple[str, ...] = (),
) -> MetricDefinition:
    return MetricDefinition(metric_id, unit, is_clock, derived_from, caveats)


METRIC_DEFINITIONS = {
    item.metric_id: item
    for item in (
        _metric("total_sleep_min", "min"),
        _metric("time_in_bed_min", "min"),
        _metric("recorded_awake_min", "min", caveats=(AWAKE_CAVEAT,)),
        _metric(
            "sleep_efficiency_pct",
            "%",
            derived_from=("total_sleep_min", "time_in_bed_min"),
            caveats=(EFFICIENCY_CAVEAT,),
        ),
        _metric("sleep_period_start_clock_min", "clock_minute", is_clock=True),
        _metric("wake_clock_min", "clock_minute", is_clock=True),
        _metric("sleep_period_midpoint_clock_min", "clock_minute", is_clock=True),
        _metric("deep_sleep_min", "min", caveats=(STAGE_CAVEAT,)),
        _metric("light_sleep_min", "min", caveats=(STAGE_CAVEAT,)),
        _metric("rem_sleep_min", "min", caveats=(STAGE_CAVEAT,)),
        _metric("wake_stage_min", "min", caveats=(STAGE_CAVEAT,)),
        _metric("steps", "steps/day"),
        _metric("sedentary_minutes", "min/day", caveats=(ACTIVITY_LEVEL_CAVEAT,)),
        _metric("lightly_active_minutes", "min/day", caveats=(ACTIVITY_LEVEL_CAVEAT,)),
        _metric("fairly_active_minutes", "min/day", caveats=(ACTIVITY_LEVEL_CAVEAT,)),
        _metric("very_active_minutes", "min/day", caveats=(ACTIVITY_LEVEL_CAVEAT,)),
        _metric(
            "active_minutes",
            "min/day",
            derived_from=("fairly_active_minutes", "very_active_minutes"),
            caveats=(ACTIVITY_LEVEL_CAVEAT,),
        ),
        _metric("activity_calories", "kcal/day", caveats=(ACTIVITY_CALORIE_CAVEAT,)),
        _metric(
            "activity_classified_minutes",
            "min/day",
            caveats=(ACTIVITY_LEVEL_CAVEAT,),
        ),
        _metric("daily_heart_rate_mean", "beats/min", caveats=(DAILY_HEART_RATE_CAVEAT,)),
        _metric("daily_heart_rate_min", "beats/min", caveats=(DAILY_HEART_RATE_CAVEAT,)),
        _metric("daily_heart_rate_max", "beats/min", caveats=(DAILY_HEART_RATE_CAVEAT,)),
        _metric(
            "heart_rate_valid_samples",
            "samples/day",
            caveats=(HEART_RATE_COVERAGE_CAVEAT,),
        ),
        _metric(
            "heart_rate_unique_timestamps",
            "timestamps/day",
            caveats=(HEART_RATE_COVERAGE_CAVEAT,),
        ),
        _metric(
            "heart_rate_unique_minutes",
            "min/day",
            caveats=(HEART_RATE_COVERAGE_CAVEAT,),
        ),
        _metric(
            "heart_rate_observed_span_min",
            "min/day",
            caveats=(HEART_RATE_COVERAGE_CAVEAT,),
        ),
        _metric(
            "heart_rate_coverage_rate_24h",
            "proportion",
            caveats=(HEART_RATE_COVERAGE_CAVEAT,),
        ),
        _metric(
            "heart_rate_sampling_density",
            "samples/observed_minute",
            caveats=(HEART_RATE_COVERAGE_CAVEAT,),
        ),
        _metric(
            "hr_zone_out_of_range_minutes",
            "min/day",
            caveats=(HEART_RATE_ZONE_CAVEAT,),
        ),
        _metric(
            "hr_zone_fat_burn_minutes",
            "min/day",
            caveats=(HEART_RATE_ZONE_CAVEAT,),
        ),
        _metric(
            "hr_zone_cardio_minutes",
            "min/day",
            caveats=(HEART_RATE_ZONE_CAVEAT,),
        ),
        _metric(
            "hr_zone_peak_minutes",
            "min/day",
            caveats=(HEART_RATE_ZONE_CAVEAT,),
        ),
        _metric(
            "total_hr_zone_minutes",
            "min/day",
            caveats=(HEART_RATE_ZONE_CAVEAT,),
        ),
    )
}

SLEEP_METRICS = frozenset(
    {
        "total_sleep_min",
        "time_in_bed_min",
        "recorded_awake_min",
        "sleep_efficiency_pct",
        "sleep_period_start_clock_min",
        "wake_clock_min",
        "sleep_period_midpoint_clock_min",
        "deep_sleep_min",
        "light_sleep_min",
        "rem_sleep_min",
        "wake_stage_min",
    }
)
CLOCK_METRICS = frozenset(
    metric_id for metric_id, definition in METRIC_DEFINITIONS.items() if definition.is_clock
)


def definition(metric_id: str) -> MetricDefinition:
    return METRIC_DEFINITIONS[metric_id]

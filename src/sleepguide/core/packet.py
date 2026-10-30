"""The compact model input shared by Stage 1 and Stage 2: profile packet P with longitudinal insight."""

from __future__ import annotations

from typing import Any

GENERATION_EXCLUDED_METRICS = frozenset(
    {
        # Source stage-entry counts are structural summaries, not actionable outcomes.
        "count_deep",
        "count_light",
        "count_rem",
        "count_wake",
        # These interval aggregates remain in the audit Profile. Only the count of
        # episodes over five minutes has a compatible Evidence reference.
        "wake_episode_count",
        "longest_wake_episode_min",
        "wake_episode_total_min",
        # Duplicate or edge-component fields are retained in Profile but should not
        # compete with canonical WASO/final-wake measures in O.
        "wake_stage_min",
        "minute_after_wakeup",
        # Preserve the raw source categories in Profile; use steps or active_minutes
        # for any carefully limited activity context shown to the model.
        "sedentary_minutes",
        "lightly_active_minutes",
        "fairly_active_minutes",
        "very_active_minutes",
    }
)


def _keep(item: dict[str, Any]) -> bool:
    return item.get("metric_id") not in GENERATION_EXCLUDED_METRICS


def generation_profile_packet(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep case context once; leave longitudinal values to the Insight packet.

    The canonical Profile remains in the Stage-1 artifact.  This model-facing profile
    carries coverage/reliability metadata for every Evidence-addressable metric plus the
    two analyses that are not duplicated by the five-view Insight: weekday/weekend
    comparisons and daily activity associations.
    """

    metric_coverage = [
        {
            key: item.get(key)
            for key in (
                "metric_id",
                "unit",
                "n",
                "span_days",
                "sampling_density_pct",
                "window_tier",
                "caveats",
            )
        }
        for item in profile.get("longitudinal_summaries", [])
        if _keep(item)
    ]
    weekday_weekend = [
        item for item in profile.get("weekday_weekend_comparisons", []) if _keep(item)
    ]
    associations = [
        item
        for item in profile.get("daily_context_associations", [])
        if item.get("context_metric_id") not in GENERATION_EXCLUDED_METRICS
        and item.get("sleep_metric_id") not in GENERATION_EXCLUDED_METRICS
    ]
    return {
        "case_id": profile.get("case_id"),
        "age_at_case_start_years": profile.get("age_at_case_start_years"),
        "sex": profile.get("sex"),
        "valid_main_sleep_nights": profile.get("valid_main_sleep_nights"),
        "calendar_span_days": profile.get("calendar_span_days"),
        "coverage_pct": profile.get("coverage_pct"),
        "metric_coverage": metric_coverage,
        "weekday_weekend_comparisons": weekday_weekend,
        "daily_context_associations": associations,
        "data_issues": profile.get("data_issues", []),
        "interpretation_caveats": profile.get("interpretation_caveats", []),
    }


def compact_insight(insight: dict[str, Any]) -> dict[str, Any]:
    """Preserve all five views for every metric without repeating verbose windows."""

    trajectories = []
    for item in insight.get("rolling_7_night_trajectory", []):
        windows = item.get("windows", [])
        trajectories.append(
            {
                "metric_id": item.get("metric_id"),
                "unit": item.get("unit"),
                "window_size_nights": item.get("window_size_nights"),
                "first_start_date": windows[0].get("start_date") if windows else None,
                "series": [
                    [window.get("end_date"), window.get("n"), window.get("center")]
                    for window in windows
                ],
            }
        )
    return {
        "case_id": insight.get("case_id"),
        "whole_record_summary": insight.get("whole_record_summary", []),
        "rolling_7_night_trajectory": trajectories,
        "recent_7_night_comparison": insight.get("recent_7_night_comparison", []),
        "latest_night_context": insight.get("latest_night_context", []),
        "fluctuation_points": insight.get("fluctuation_points", []),
    }


def generation_insight_packet(insight: dict[str, Any]) -> dict[str, Any]:
    packet = compact_insight(insight)
    for field in (
        "whole_record_summary",
        "rolling_7_night_trajectory",
        "recent_7_night_comparison",
        "latest_night_context",
        "fluctuation_points",
    ):
        packet[field] = [item for item in packet.get(field, []) if _keep(item)]
    return packet


def generation_case_signals(profile: dict[str, Any]) -> dict[str, bool]:
    """Expose deterministic semantic signals that are easy to miss in compact tables.

    These signals do not select Evidence or make a health judgment. They only identify
    when a derived metric contributes information that its component metrics do not each
    show clearly on their own.
    """

    comparisons = {
        item.get("metric_id"): item
        for item in profile.get("weekday_weekend_comparisons", [])
        if item.get("reportable")
    }
    summaries = {
        item.get("metric_id"): item for item in profile.get("metric_summaries", [])
    }

    def salient(metric_id: str, *, variability_multiple: float = 1.0) -> bool:
        comparison = comparisons.get(metric_id)
        summary = summaries.get(metric_id)
        difference = comparison.get("weekend_minus_weekday") if comparison else None
        mad = summary.get("mad") if summary else None
        return bool(
            difference is not None
            and mad is not None
            and abs(float(difference))
            > variability_multiple * max(float(mad), 1e-9)
        )

    onset = comparisons.get("sleep_onset_clock_min")
    final_wake = comparisons.get("final_wake_clock_min")
    onset_difference = onset.get("weekend_minus_weekday") if onset else None
    wake_difference = final_wake.get("weekend_minus_weekday") if final_wake else None
    same_direction = bool(
        onset_difference is not None
        and wake_difference is not None
        and float(onset_difference) * float(wake_difference) > 0
    )
    return {
        "weekday_weekend_whole_sleep_period_shift_is_midpoint_decisive": bool(
            salient("sleep_midpoint_clock_min", variability_multiple=2.0)
            and same_direction
            and not salient("sleep_onset_clock_min", variability_multiple=2.0)
            and not salient("final_wake_clock_min", variability_multiple=2.0)
        )
    }


def generation_model_input(
    profile: dict[str, Any],
    insight: dict[str, Any],
) -> dict[str, Any]:
    """Return the single P representation shared by Stage 1 and Stage 2."""

    return {
        "schema_version": "profile-packet-1.0",
        "profile": generation_profile_packet(profile),
        "longitudinal_insight": generation_insight_packet(insight),
        "case_signals": generation_case_signals(profile),
    }

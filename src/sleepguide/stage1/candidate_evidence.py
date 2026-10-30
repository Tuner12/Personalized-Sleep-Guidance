"""Deterministic candidate evidence pools offered to the Evidence Agent for each case."""

from __future__ import annotations

from dataclasses import dataclass

from sleepguide.core.contracts import LongitudinalInsight, SleepProfile
from sleepguide.stage1.contracts import EvidenceCard, EvidenceCardLibrary

# The source recommendation is expressed in whole hours, while the normalized daily
# fields are integer-minute estimates.  This epsilon handles only the exact unit edge
# (for example, 6 h 59 min versus 7 h); it is not a clinical margin or a replacement
# threshold.  Any wider practical-equivalence band requires separate expert validation.
_REFERENCE_UNIT_EPSILON_MIN = 1.0
_SLEEP_ONSET = "sleep_onset_clock_min"
_FINAL_WAKE = "final_wake_clock_min"
_SLEEP_MIDPOINT = "sleep_midpoint_clock_min"


@dataclass(frozen=True)
class EvidenceCandidatePools:
    """Deterministic candidate pools separated by observable applicability.

    A direct candidate has no unobserved condition in its card.  A conditional
    candidate has a case-matched outcome, but its exposure, preference, symptom, or
    care context is not present in the current input and must remain explicit as an
    ``if`` condition.  Pool membership is not final Evidence selection.
    """

    direct: tuple[EvidenceCard, ...]
    conditional: tuple[EvidenceCard, ...]

    @property
    def all(self) -> tuple[EvidenceCard, ...]:
        return (*self.direct, *self.conditional)


def _whole(profile: SleepProfile, metric_id: str) -> float | None:
    item = next((x for x in profile.metric_summaries if x.metric_id == metric_id), None)
    return item.median if item is not None else None


def _summary(profile: SleepProfile, metric_id: str):
    return next((x for x in profile.metric_summaries if x.metric_id == metric_id), None)


def _recent(insight: LongitudinalInsight, metric_id: str) -> float | None:
    item = next((x for x in insight.recent_7_night_comparison if x.metric_id == metric_id), None)
    return (
        item.recent_7_nights.center
        if item is not None and item.recent_7_nights is not None
        else None
    )


def _previous(insight: LongitudinalInsight, metric_id: str) -> float | None:
    item = next((x for x in insight.recent_7_night_comparison if x.metric_id == metric_id), None)
    return (
        item.previous_7_nights.center
        if item is not None and item.previous_7_nights is not None
        else None
    )


def _longitudinal_direction(profile: SleepProfile, metric_id: str) -> float | None:
    item = next((x for x in profile.longitudinal_summaries if x.metric_id == metric_id), None)
    if (
        item is None
        or item.recent_minus_earlier is None
        or item.comparison_direction_agrees_with_slope is not True
    ):
        return None
    return item.recent_minus_earlier


def _salient_longitudinal_direction(
    profile: SleepProfile, metric_id: str, *, variability_multiple: float = 1.0
) -> float | None:
    """Return a direction only when it exceeds the case's own typical variability.

    This is an internal retrieval-noise gate, not a health threshold or a reader target.
    """

    direction = _longitudinal_direction(profile, metric_id)
    metric_summary = _summary(profile, metric_id)
    variability = metric_summary.mad if metric_summary is not None else None
    if direction is None or variability is None:
        return None
    if abs(direction) <= variability_multiple * max(variability, 1e-9):
        return None
    return direction


def _salient_recent_change(
    profile: SleepProfile,
    insight: LongitudinalInsight,
    metric_id: str,
) -> float | None:
    """Return recent-minus-previous only when larger than personal variability."""

    recent = _recent(insight, metric_id)
    previous = _previous(insight, metric_id)
    summary = _summary(profile, metric_id)
    if (
        recent is None
        or previous is None
        or summary is None
        or summary.mad is None
    ):
        return None
    change = recent - previous
    if abs(change) <= max(summary.mad, 1e-9):
        return None
    return change


def _reportable_weekpart(profile: SleepProfile, metric_id: str):
    return next(
        (
            item
            for item in profile.weekday_weekend_comparisons
            if item.metric_id == metric_id and item.reportable
        ),
        None,
    )


def _salient_weekpart(
    profile: SleepProfile,
    metric_id: str,
    *,
    variability_multiple: float = 1.0,
):
    """Require a calendar-part contrast larger than this case's usual variability."""

    comparison = _reportable_weekpart(profile, metric_id)
    summary = next((x for x in profile.metric_summaries if x.metric_id == metric_id), None)
    if (
        comparison is None
        or comparison.weekend_minus_weekday is None
        or summary is None
        or summary.mad is None
    ):
        return None
    if abs(comparison.weekend_minus_weekday) <= variability_multiple * max(
        summary.mad, 1e-9
    ):
        return None
    return comparison


def _salient_stage_direction(profile: SleepProfile, metric_id: str) -> float | None:
    """Use a stricter case-relative gate for descriptive sleep-stage changes."""

    direction = _longitudinal_direction(profile, metric_id)
    summary = _summary(profile, metric_id)
    if direction is None or summary is None or summary.mad is None or summary.mad <= 0:
        return None
    return direction if abs(direction) > 2 * summary.mad else None


def _reportable_context_association(
    profile: SleepProfile, context_metric_id: str, sleep_metric_id: str
):
    return next(
        (
            item
            for item in profile.daily_context_associations
            if item.context_metric_id == context_metric_id
            and item.sleep_metric_id == sleep_metric_id
            and item.reportable
        ),
        None,
    )


def _duration_reference_key(age: int | None) -> str | None:
    if age is not None and 18 <= age <= 64:
        return "adults-18-to-64-usually-need-at-least-7-hours"
    if age is not None and age >= 65:
        return "adults-65-plus-usually-need-at-least-7-hours"
    return None


def _waso_reference(age: int | None) -> tuple[str, float] | None:
    if age is not None and 18 <= age <= 64:
        return "waso-quality-reference-for-adults-18-to-64", 20.0
    if age is not None and age >= 65:
        return "waso-quality-reference-for-adults-65-plus", 30.0
    return None


def _long_wake_reference(age: int | None) -> tuple[str, float] | None:
    if age is not None and 18 <= age <= 64:
        return "long-wake-episode-reference-for-adults-18-to-64", 1.0
    if age is not None and age >= 65:
        return "long-wake-episode-reference-for-adults-65-plus", 2.0
    return None


def candidate_evidence_pools(
    profile: SleepProfile,
    insight: LongitudinalInsight,
    library: EvidenceCardLibrary,
) -> EvidenceCandidatePools:
    """Return broad deterministic pools without selecting an action or exposure."""

    keys: set[str] = set()
    age = profile.age_at_case_start_years
    whole_sleep = _whole(profile, "total_sleep_min")
    recent_sleep = _recent(insight, "total_sleep_min")
    recent_tib = _recent(insight, "time_in_bed_min")
    recent_efficiency = _recent(insight, "sleep_efficiency_pct")
    whole_waso = _whole(profile, "waso_min")
    recent_waso = _recent(insight, "waso_min")
    whole_latency = _whole(profile, "minute_to_fall_asleep")
    recent_latency = _recent(insight, "minute_to_fall_asleep")
    whole_long_wakes = _whole(profile, "wake_episode_over_5_min_count")
    recent_long_wakes = _recent(insight, "wake_episode_over_5_min_count")

    def clearly_below_duration_reference(value: float | None) -> bool:
        return value is not None and value < 420 - _REFERENCE_UNIT_EPSILON_MIN

    def at_duration_reference(value: float | None) -> bool:
        return value is not None and value >= 420 - _REFERENCE_UNIT_EPSILON_MIN

    recent_short_pattern = clearly_below_duration_reference(recent_sleep)
    low_efficiency = recent_efficiency is not None and recent_efficiency < 85
    limited_opportunity = (
        recent_short_pattern
        and recent_tib is not None
        and clearly_below_duration_reference(recent_tib)
    )
    if recent_short_pattern:
        duration_reference_key = _duration_reference_key(age)
        if duration_reference_key:
            keys.add(duration_reference_key)
        if limited_opportunity:
            keys.add("allow-more-time-for-sleep")
        elif recent_tib is not None and at_duration_reference(recent_tib):
            keys.add("time-in-bed-is-not-the-same-as-sleep-duration")

    waso_reference = _waso_reference(age)
    persistent_elevated_waso_crossing = bool(
        waso_reference is not None
        and whole_waso is not None
        and recent_waso is not None
        and whole_waso > waso_reference[1]
        and recent_waso > waso_reference[1]
    )
    persistent_long_latency = bool(
        whole_latency is not None
        and recent_latency is not None
        and whole_latency > 30
        and recent_latency > 30
    )
    long_wake_reference = _long_wake_reference(age)
    persistent_long_wake_episodes_crossing = bool(
        long_wake_reference is not None
        and whole_long_wakes is not None
        and recent_long_wakes is not None
        and whole_long_wakes > long_wake_reference[1]
        and recent_long_wakes > long_wake_reference[1]
    )
    waso_direction = _salient_longitudinal_direction(profile, "waso_min")
    waso_recent_change = _salient_recent_change(profile, insight, "waso_min")
    long_wake_direction = _salient_longitudinal_direction(
        profile, "wake_episode_over_5_min_count"
    )
    long_wake_recent_change = _salient_recent_change(
        profile, insight, "wake_episode_over_5_min_count"
    )
    # Crossing a population quality reference is only case Evidence when it changes
    # the decision: recent efficiency is also low, or the same metric clearly worsened
    # relative to this person's record. This prevents a stable automated estimate from
    # becoming a default recommendation trigger.
    persistent_elevated_waso_raw = bool(
        persistent_elevated_waso_crossing
        and (
            low_efficiency
            or (waso_direction is not None and waso_direction > 0)
            or (waso_recent_change is not None and waso_recent_change > 0)
            or (
                recent_short_pattern
                and recent_tib is not None
                and at_duration_reference(recent_tib)
            )
        )
    )
    persistent_long_wake_episodes_raw = bool(
        persistent_long_wake_episodes_crossing
        and (
            low_efficiency
            or (long_wake_direction is not None and long_wake_direction > 0)
            or (
                long_wake_recent_change is not None
                and long_wake_recent_change > 0
            )
            or (
                recent_short_pattern
                and recent_tib is not None
                and at_duration_reference(recent_tib)
            )
        )
    )
    # WASO and the count of longer wake episodes describe different aspects of
    # continuity: total wake time versus the number of longer episodes. They may both
    # be candidates, but they must never be counted as independent clinical triggers.
    persistent_elevated_waso = persistent_elevated_waso_raw
    persistent_long_wake_episodes = persistent_long_wake_episodes_raw

    repeated_continuity_issue = (
        low_efficiency
        or persistent_elevated_waso
        or persistent_long_latency
        or persistent_long_wake_episodes
        or (waso_direction is not None and waso_direction > 0)
        or (waso_recent_change is not None and waso_recent_change > 0)
    )
    if (
        recent_short_pattern
        and recent_tib is not None
        and at_duration_reference(recent_tib)
        and (
            persistent_elevated_waso_crossing
            or persistent_long_wake_episodes_crossing
            or low_efficiency
        )
    ):
        keys.add("caffeine-close-to-sleep-may-disrupt-sleep")
    if low_efficiency:
        keys.add("sleep-efficiency-below-85-percent-may-reflect-poor-continuity")
        if recent_tib is not None and recent_sleep is not None:
            keys.add("time-in-bed-is-not-the-same-as-sleep-duration")
    if persistent_elevated_waso and waso_reference is not None:
        keys.add(waso_reference[0])
    if persistent_long_latency:
        keys.add("sleep-latency-quality-reference-for-adults")
        keys.add("a-quiet-wind-down-may-support-sleep")
    if persistent_long_wake_episodes and long_wake_reference is not None:
        keys.add(long_wake_reference[0])
    if repeated_continuity_issue:
        # Aggregate continuity measures cannot distinguish caffeine, alcohol, and
        # nighttime noise.  Do not make those interchangeable exposures compete in
        # the Evidence Agent.  Caffeine remains eligible above only when repeated short
        # duration supplies its more specific supported outcome.
        whole_efficiency = _whole(profile, "sleep_efficiency_pct")
        # Professional care is never a direct consequence of crossing a duration
        # reference.  Offer it only as a conditional boundary when both whole-record
        # and recent continuity are repeatedly poor; symptoms and daytime effects are
        # still unobserved and must remain explicit.
        persistent_low_efficiency = bool(
            whole_efficiency is not None and whole_efficiency < 85 and low_efficiency
        )
        additional_continuity_signal = bool(
            persistent_elevated_waso
            or persistent_long_latency
            or persistent_long_wake_episodes
        )
        if persistent_low_efficiency and additional_continuity_signal:
            keys.add("persistent-sleep-difficulty-needs-clinical-context")
        activity_efficiency = _reportable_context_association(
            profile, "active_minutes", "sleep_efficiency_pct"
        )
        if (
            activity_efficiency is not None
            and activity_efficiency.high_minus_low_sleep is not None
            and activity_efficiency.high_minus_low_sleep > 0
            and low_efficiency
        ):
            keys.add("regular-daytime-movement-may-support-sleep-efficiency")

    # Timing guidance needs a stronger within-person signal than descriptive reporting.
    # Two personal MADs is a retrieval-noise gate, not a universal health threshold or
    # reader-facing target.
    start_weekpart = _salient_weekpart(
        profile, _SLEEP_ONSET, variability_multiple=2.0
    )
    wake_weekpart = _salient_weekpart(
        profile, _FINAL_WAKE, variability_multiple=2.0
    )
    midpoint_weekpart = _salient_weekpart(
        profile, _SLEEP_MIDPOINT, variability_multiple=2.0
    )
    salient_duration_weekpart = _salient_weekpart(profile, "total_sleep_min")
    raw_start_weekpart = _reportable_weekpart(profile, _SLEEP_ONSET)
    raw_wake_weekpart = _reportable_weekpart(profile, _FINAL_WAKE)
    coherent_start_wake = bool(
        start_weekpart is not None
        and wake_weekpart is not None
        and start_weekpart.weekend_minus_weekday * wake_weekpart.weekend_minus_weekday > 0
    )
    # Midpoint captures a coherent shift of the full sleep period when onset and final
    # wake move in the same direction but either component alone is noisy relative to
    # its own variability. Opposing onset/wake changes are duration changes, not a
    # coherent timing shift, and must not be routed through midpoint alone.
    coherent_midpoint_shift = bool(
        midpoint_weekpart is not None
        and raw_start_weekpart is not None
        and raw_wake_weekpart is not None
        and raw_start_weekpart.weekend_minus_weekday is not None
        and raw_wake_weekpart.weekend_minus_weekday is not None
        and raw_start_weekpart.weekend_minus_weekday
        * raw_wake_weekpart.weekend_minus_weekday
        > 0
    )
    later_start_with_shorter_sleep = bool(
        start_weekpart is not None
        and raw_wake_weekpart is not None
        and salient_duration_weekpart is not None
        and start_weekpart.weekend_minus_weekday is not None
        and raw_wake_weekpart.weekend_minus_weekday is not None
        and salient_duration_weekpart.weekend_minus_weekday is not None
        and start_weekpart.weekend_minus_weekday > 0
        and salient_duration_weekpart.weekend_minus_weekday < 0
        and abs(raw_wake_weekpart.weekend_minus_weekday)
        < abs(start_weekpart.weekend_minus_weekday)
    )
    timing_weekpart = (
        coherent_start_wake
        or coherent_midpoint_shift
        or (start_weekpart is not None and salient_duration_weekpart is not None)
        or later_start_with_shorter_sleep
    )
    if timing_weekpart:
        keys.update(
            {
                "make-sleep-timing-more-consistent",
                "sleep-timing-alone-does-not-confirm-circadian-phase",
            }
        )
        sleep_weekpart = salient_duration_weekpart
        if (
            sleep_weekpart is not None
            and sleep_weekpart.weekday_center is not None
            and sleep_weekpart.weekend_minus_weekday is not None
            and clearly_below_duration_reference(sleep_weekpart.weekday_center)
            and sleep_weekpart.weekend_minus_weekday > 0
        ):
            keys.add("preserve-recovery-sleep-when-weekday-sleep-is-short")
            time_in_bed_weekpart = _reportable_weekpart(profile, "time_in_bed_min")
            if (
                time_in_bed_weekpart is not None
                and time_in_bed_weekpart.weekday_center is not None
                and clearly_below_duration_reference(time_in_bed_weekpart.weekday_center)
            ):
                keys.add("allow-more-time-for-sleep")
            duration_reference_key = _duration_reference_key(age)
            if duration_reference_key:
                keys.add(duration_reference_key)

    later_start = _salient_longitudinal_direction(profile, _SLEEP_ONSET)
    later_wake = _salient_longitudinal_direction(profile, _FINAL_WAKE)
    later_onset = bool(later_start is not None and later_start > 0)
    whole_period_later = bool(
        later_onset and later_wake is not None and later_wake > 0
    )
    if later_onset:
        keys.add("sleep-timing-alone-does-not-confirm-circadian-phase")
        if whole_period_later:
            keys.add("unwanted-sleep-timing-may-need-clinical-review")
        else:
            keys.add("late-light-or-screen-use-may-delay-sleep")

    stage_direction = any(
        _salient_stage_direction(profile, metric_id) not in (None, 0)
        for metric_id in ("deep_sleep_min", "rem_sleep_min")
    )
    if stage_direction:
        keys.add("sleep-stage-estimates-are-not-action-targets")

    # A full recent seven-night period at the reference is enough to make maintenance
    # available even when earlier nights pull the whole-record median down.  The Evidence
    # Agent still decides whether this is genuine recovery rather than hiding an unresolved
    # earlier pattern.
    duration_currently_adequate = bool(at_duration_reference(recent_sleep))
    waso_currently_adequate = not persistent_elevated_waso
    long_wakes_currently_adequate = not persistent_long_wake_episodes
    latency_currently_adequate = bool(recent_latency is None or recent_latency <= 30)
    continuity_currently_adequate = bool(
        recent_efficiency is not None
        and recent_efficiency >= 85
        and waso_currently_adequate
        and long_wakes_currently_adequate
        and latency_currently_adequate
    )
    no_actionable_timing_change = not timing_weekpart and not later_onset
    if (
        duration_currently_adequate
        and continuity_currently_adequate
        and no_actionable_timing_change
    ):
        keys.add("maintain-a-stable-adequate-sleep-pattern")

    by_key = {card.evidence_key: card for card in library.cards if card.runtime_enabled}
    duration_action_keys = {
        key
        for key in keys
        if key in by_key
        and by_key[key].next_step
        and "total_sleep_min" in by_key[key].evaluate_with
    }
    persistent_short_sleep = bool(
        recent_short_pattern
        and whole_sleep is not None
        and clearly_below_duration_reference(whole_sleep)
    )
    if (
        persistent_short_sleep
        and recent_tib is not None
        and at_duration_reference(recent_tib)
        and not duration_action_keys
    ):
        keys.add("persistent-short-sleep-needs-clinical-context")

    continuity_outcomes: set[str] = set()
    if low_efficiency:
        continuity_outcomes.add("sleep_efficiency_pct")
    if (
        persistent_elevated_waso
        or (waso_direction is not None and waso_direction > 0)
        or (waso_recent_change is not None and waso_recent_change > 0)
    ):
        continuity_outcomes.add("waso_min")
    if persistent_long_latency:
        continuity_outcomes.add("minute_to_fall_asleep")
    if persistent_long_wake_episodes:
        continuity_outcomes.add("wake_episode_over_5_min_count")
    has_continuity_action = any(
        by_key[key].next_step
        and continuity_outcomes.intersection(by_key[key].evaluate_with)
        for key in keys
        if key in by_key
    )
    if repeated_continuity_issue and not has_continuity_action:
        # A continuity-only record cannot justify choosing among unmeasured caffeine,
        # alcohol, noise, or bedroom factors. The only honest next step is conditional
        # on persistent difficulty or daytime effects, which the current input does not
        # observe.
        keys.add("persistent-sleep-difficulty-needs-clinical-context")
    cards = tuple(
        card for card in library.cards if card.evidence_key in keys and card.evidence_key in by_key
    )
    return EvidenceCandidatePools(
        direct=tuple(card for card in cards if not card.condition),
        conditional=tuple(card for card in cards if card.condition),
    )

import json
from pathlib import Path
from datetime import date, timedelta

from sleepguide.core.contracts import (
    DailyObservation,
    LongitudinalInsight,
    LongitudinalSummary,
    MetricSummary,
    SleepProfile,
    WeekpartComparison,
    WindowTier,
)
from sleepguide.core.recommendation import (
    AnchorOrigin,
    AnchorStatistic,
    CaseMetricAnchor,
    ReaderRecommendation,
    _anchor_value_is_visible,
)
from sleepguide.core.packet import generation_case_signals
from sleepguide.stage1 import (
    BlindReverseReview,
    CaseObservation,
    EvidenceSelection,
    RecommendationDraft,
    ReverseIssueType,
    SectionGrounding,
    load_generation_card_library,
    writer_card,
)
from sleepguide.stage1.agents import (
    _EXPLICIT_CONDITION_LANGUAGE,
    _INTERNAL_READER_LANGUAGE,
    _canonicalize_anchor_origin,
    _filter_unmaterialized_reference_recoveries,
    _normalize_evidence_only_review,
    _reader_sentence_style_issues,
    polish_recommendation,
    validate_observation,
    validate_recommendation,
    validate_selection,
)
from sleepguide.stage1.numeric import _insight_numeric_spans
from sleepguide.stage1.candidate_evidence import candidate_evidence_pools
from sleepguide.util import content_hash


def test_reader_language_patterns_reject_indirect_or_formulaic_links() -> None:
    import re

    assert re.search(
        _INTERNAL_READER_LANGUAGE["indirect clinical next-step wording"],
        "Persistent short sleep would be a reason to speak with a doctor.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["formulaic therefore connector"],
        "Sleep was short; therefore, change the schedule.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["repeated while linkage"],
        "Sleep increased while awake time fell while timing stayed similar.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["compressed even-as linkage"],
        "Sleep became shorter even as awake time decreased.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["abstract awake-time trend wording"],
        "Time awake after falling asleep decreased across the recent nights.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["awkward lower-amount comparison"],
        "Seven hours is a usual lower amount associated with good sleep quality.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["unnecessary one-night action reversal"],
        "Do not change it solely because of one unusual night.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["indirect monitoring judgment"],
        "Time awake after falling asleep remains worth watching.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["invented routine linkage"],
        "Keep the parts of your current routine that coincided with these gains.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["unsupported absence of concern"],
        "No repeated concern is identified in this pattern.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["awkward unequal-duration wording"],
        "Sleep was not equally long.",
        re.I,
    )
    assert re.search(
        _INTERNAL_READER_LANGUAGE["indirect outcome verdict"],
        "A repeated increase would show that the added time is helping.",
        re.I,
    )


def test_reverse_discards_non_evidence_opinions() -> None:
    recommendation = ReaderRecommendation(
        section_1="Summary text.",
        section_2="Analysis text.",
        section_3="Recommendation text.",
        section_4="Follow-up text.",
    )
    reasoning_issue = {
        "issue_type": ReverseIssueType.REASONING,
        "section": "section_2",
        "reason": "The counterpoint could be explained more fully.",
        "repair_instruction": "Add another explanation.",
    }
    review = BlindReverseReview.model_validate(
        {
            "case_id": "wrong_case",
            "reproduced_recommendation": recommendation.model_dump(mode="json"),
            "section_1": {
                "section_text": "Changed summary.",
                "recovered_evidence_keys": [],
                "issues": [],
            },
            "section_2": {
                "section_text": "Analysis text.",
                "recovered_evidence_keys": ["some-card"],
                "issues": [reasoning_issue],
            },
            "section_3": {
                "section_text": "Recommendation text.",
                "recovered_evidence_keys": [],
                "issues": [],
            },
            "section_4": {
                "section_text": "Follow-up text.",
                "recovered_evidence_keys": [],
                "issues": [],
            },
            "verdict": "revise",
        }
    )

    normalized = _normalize_evidence_only_review(
        review,
        case_id="case_stage1",
        recommendation=recommendation,
    )

    assert normalized.case_id == "case_stage1"
    assert normalized.reproduced_recommendation == recommendation
    assert normalized.section_1.section_text == recommendation.section_1
    assert normalized.section_2.recovered_evidence_keys == ["some-card"]
    assert normalized.section_2.issues == []
    assert normalized.verdict == "pass"


def test_reverse_does_not_recover_reference_from_personal_metric_alone() -> None:
    recommendation = ReaderRecommendation(
        section_1="Summary text.",
        section_2="Your recent sleep efficiency was 82.6%.",
        section_3="Recommendation text.",
        section_4="Follow-up text.",
    )
    review = BlindReverseReview.model_validate(
        {
            "case_id": "case_stage1",
            "reproduced_recommendation": recommendation.model_dump(mode="json"),
            "section_1": {
                "section_text": recommendation.section_1,
                "recovered_evidence_keys": [],
                "issues": [],
            },
            "section_2": {
                "section_text": recommendation.section_2,
                "recovered_evidence_keys": [
                    "sleep-efficiency-below-85-percent-may-reflect-poor-continuity"
                ],
                "issues": [],
            },
            "section_3": {
                "section_text": recommendation.section_3,
                "recovered_evidence_keys": [],
                "issues": [],
            },
            "section_4": {
                "section_text": recommendation.section_4,
                "recovered_evidence_keys": [],
                "issues": [],
            },
            "verdict": "pass",
        }
    )

    filtered = _filter_unmaterialized_reference_recoveries(
        review,
        recommendation=recommendation.model_dump(mode="json"),
        library=load_generation_card_library(),
    )

    assert filtered.section_2.recovered_evidence_keys == []


def test_reference_anchor_visibility_accepts_natural_threshold_phrasing() -> None:
    assert _anchor_value_is_visible("Adults generally need 7 or more hours of sleep.", 420, "min")
    assert _anchor_value_is_visible("At least 7 hours is a useful general comparison.", 420, "min")


def test_conditional_language_does_not_require_if_at_sentence_start() -> None:
    accepted = (
        "If caffeine is used near bedtime, move it earlier.",
        "Move caffeine earlier when it is used near bedtime.",
        "Consider this change only when later timing is unwanted.",
        "Keep the change as long as sleep duration remains stable.",
    )
    assert all(_EXPLICIT_CONDITION_LANGUAGE.search(text) for text in accepted)


def test_reader_style_check_detects_clause_and_metric_overload() -> None:
    overloaded = (
        "Sleep duration fell while time in bed decreased, but sleep efficiency remained "
        "similar and time awake after falling asleep increased."
    )
    issues = _reader_sentence_style_issues(overloaded)
    assert "multiple contrast links in one sentence" in issues
    assert "overloaded multi-metric sentence" in issues
    assert not _reader_sentence_style_issues(
        "Sleep duration fell. Time in bed also became shorter."
    )


def test_focused_polish_restores_immutable_grounding() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id=profile.case_id,
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep and time in bed were repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    source = RecommendationDraft(
        case_id=profile.case_id,
        section_1="Sleep was usually short.",
        section_2="Sleep and time in bed were repeatedly short.",
        section_3=(
            "**Make more time for sleep**\n"
            "When practical, leave more time for sleep."
        ),
        section_4="Over the next several nights, see whether sleep lasts longer.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )
    changed = source.model_copy(
        update={
            "section_1_grounding": SectionGrounding(evidence_keys=[key]),
            "section_3": (
                "**An invented polish title**\n"
                "When practical, leave more time for sleep."
            ),
        }
    )

    class StubClient:
        def structured(self, **_: object) -> RecommendationDraft:
            return changed

    restored, issues = polish_recommendation(
        StubClient(),  # type: ignore[arg-type]
        profile,
        insight,
        selection,
        library,
        source,
    )
    assert restored.section_1_grounding == source.section_1_grounding
    assert restored.section_3.startswith("**Make more time for sleep**")
    assert "language polish changed Grounding" not in " ".join(issues)


def test_activity_follow_up_cannot_claim_movement_is_helping() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "regular-daytime-movement-may-support-sleep-efficiency"
    selection = EvidenceSelection(
        case_id=profile.case_id,
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep efficiency tended to be higher after more active days.",
                "matched_outcomes": ["sleep_efficiency_pct"],
            }
        ],
        selection_summary="Maintain usual daytime movement.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id=profile.case_id,
        section_1="Sleep efficiency varied across the record.",
        section_2="Sleep efficiency tended to be higher after more active days.",
        section_3=(
            "**Keep daytime movement regular**\n"
            "When safe and practical, keep your usual daytime movement."
        ),
        section_4=(
            "See whether sleep efficiency is higher after more active days. "
            "If it is, regular daytime movement is helping your sleep efficiency."
        ),
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["sleep_efficiency_pct"]
        ),
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "causal benefit" in " ".join(issues)


# Longitudinal insights of the synthetic profiles below (tests/fixtures/insights.json, keyed
# by the profile's content hash).
_INSIGHTS = json.loads((Path(__file__).parent / "fixtures/insights.json").read_text())


def build_longitudinal_insight(profile: SleepProfile) -> LongitudinalInsight:
    return LongitudinalInsight.model_validate(_INSIGHTS[content_hash(profile)])


def _profile() -> SleepProfile:
    return SleepProfile(
        case_id="case_stage1",
        age_at_case_start_years=42,
        valid_main_sleep_nights=14,
        calendar_span_days=14,
        coverage_pct=100,
        daily_observations=[
            DailyObservation(
                sleep_date=date(2026, 1, 1) + timedelta(days=index - 1),
                day_index=index,
                day_of_week=(date(2026, 1, 1) + timedelta(days=index - 1)).strftime("%A"),
                is_weekend=(date(2026, 1, 1) + timedelta(days=index - 1)).weekday() >= 5,
                metrics={"total_sleep_min": 360.0, "time_in_bed_min": 390.0},
            )
            for index in range(1, 15)
        ],
        metric_summaries=[
            MetricSummary(metric_id="total_sleep_min", unit="min", n=14, median=360),
            MetricSummary(metric_id="time_in_bed_min", unit="min", n=14, median=390),
        ],
    )


def _observation() -> CaseObservation:
    anchor = CaseMetricAnchor(
        metric_id="total_sleep_min",
        origin=AnchorOrigin.AGGREGATE,
        statistic=AnchorStatistic.MEDIAN,
        value=360,
        unit="min",
    )
    return CaseObservation(
        case_id="case_stage1",
        primary_observation="Sleep was repeatedly short.",
        supporting_observations=[
            "Sleep was typically about 6 hours across the record."
        ],
        context_observation=(
            "Time in bed was also limited rather than consistently long."
        ),
        relevant_metric_ids=["total_sleep_min", "time_in_bed_min"],
        anchors=[anchor],
    )


def test_compact_observation_rejects_questions_and_duplicate_fields() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    observation = _observation().model_copy(
        update={
            "primary_observation": "Was sleep repeatedly short?",
            "supporting_observations": ["Was sleep repeatedly short?"],
            "context_observation": "Short sleep was the main concern.",
        }
    )
    issues = validate_observation(observation, profile, insight)
    joined = " ".join(issues)
    assert "repeats the same fact" in joined
    assert "not an interpretive question" in joined
    assert "main concern or problem" in joined


def _stable_profile(
    *,
    sleep_min: float,
    time_in_bed_min: float,
    efficiency_pct: float,
    waso_min: float | None = None,
    latency_min: float | None = None,
    age: int = 42,
) -> SleepProfile:
    optional_metrics = {
        **({"waso_min": waso_min} if waso_min is not None else {}),
        **({"minute_to_fall_asleep": latency_min} if latency_min is not None else {}),
    }
    observations = [
        DailyObservation(
            sleep_date=date(2026, 1, 1) + timedelta(days=index - 1),
            day_index=index,
            day_of_week=(date(2026, 1, 1) + timedelta(days=index - 1)).strftime("%A"),
            is_weekend=(date(2026, 1, 1) + timedelta(days=index - 1)).weekday() >= 5,
            metrics={
                "total_sleep_min": sleep_min,
                "time_in_bed_min": time_in_bed_min,
                "sleep_efficiency_pct": efficiency_pct,
                **optional_metrics,
            },
        )
        for index in range(1, 15)
    ]
    return SleepProfile(
        case_id="case_router",
        age_at_case_start_years=age,
        valid_main_sleep_nights=14,
        calendar_span_days=14,
        coverage_pct=100,
        daily_observations=observations,
        metric_summaries=[
            MetricSummary(metric_id="total_sleep_min", unit="min", n=14, median=sleep_min, mad=0),
            MetricSummary(
                metric_id="time_in_bed_min",
                unit="min",
                n=14,
                median=time_in_bed_min,
                mad=0,
            ),
            MetricSummary(
                metric_id="sleep_efficiency_pct",
                unit="%",
                n=14,
                median=efficiency_pct,
                mad=0,
            ),
            *(
                [
                    MetricSummary(
                        metric_id="waso_min",
                        unit="min",
                        n=14,
                        median=waso_min,
                        mad=0,
                    )
                ]
                if waso_min is not None
                else []
            ),
            *(
                [
                    MetricSummary(
                        metric_id="minute_to_fall_asleep",
                        unit="min",
                        n=14,
                        median=latency_min,
                        mad=0,
                    )
                ]
                if latency_min is not None
                else []
            ),
        ],
    )


def test_generation_library_collapses_source_dossiers_into_writer_cards() -> None:
    library = load_generation_card_library()
    assert len(library.cards) == 26
    assert "consumer-tracker-minimum-nights" not in {card.evidence_key for card in library.cards}
    light = next(
        card
        for card in library.cards
        if card.evidence_key == "late-light-or-screen-use-may-delay-sleep"
    )
    assert set(light.source_dossier_ids) == {
        "conditional-light-and-screen-adjustment",
        "light-timing-sleep-review",
        "nsf-healthy-sleep-behaviors",
    }
    stage = next(
        card
        for card in library.cards
        if card.evidence_key == "sleep-stage-estimates-are-not-action-targets"
    )
    assert stage.next_step is None
    assert stage.evaluate_with == []
    timing_care = next(
        card
        for card in library.cards
        if card.evidence_key == "unwanted-sleep-timing-may-need-clinical-review"
    )
    melatonin = next(
        card
        for card in library.cards
        if card.evidence_key == "melatonin-timing-needs-professional-guidance"
    )
    assert "clinician-guided-melatonin" in timing_care.source_dossier_ids
    assert timing_care.runtime_enabled is True
    assert melatonin.runtime_enabled is False
    persistent_short = next(
        card
        for card in library.cards
        if card.evidence_key == "persistent-short-sleep-needs-clinical-context"
    )
    assert persistent_short.condition is not None
    assert "persistent-short-sleep-clinical-evaluation" in persistent_short.source_dossier_ids
    assert "melatonin" not in persistent_short.next_step.lower()
    disabled_context_cards = {
        "alcohol-close-to-sleep-may-disrupt-continuity",
        "nighttime-noise-may-disrupt-sleep",
    }
    assert all(
        not card.runtime_enabled
        for card in library.cards
        if card.evidence_key in disabled_context_cards
    )


def test_action_cards_have_small_globally_unique_controlled_title_sets() -> None:
    library = load_generation_card_library()
    action_cards = [card for card in library.cards if card.next_step]
    all_titles = [title for card in action_cards for title in card.allowed_action_titles]

    assert action_cards
    assert all(2 <= len(card.allowed_action_titles) <= 4 for card in action_cards)
    assert len(all_titles) == len(set(all_titles))


def test_writer_card_exposes_titles_without_source_audit_fields() -> None:
    library = load_generation_card_library()
    card = next(
        card for card in library.cards if card.evidence_key == "allow-more-time-for-sleep"
    )
    rendered = writer_card(card)

    assert rendered["canonical_action_title"] == card.canonical_action_title
    assert rendered["allowed_action_titles"] == card.allowed_action_titles
    assert "source_dossier_ids" not in rendered


def test_unambiguous_anchor_origin_is_canonicalized() -> None:
    anchor = CaseMetricAnchor(
        metric_id="total_sleep_min",
        origin=AnchorOrigin.INSIGHT,
        statistic=AnchorStatistic.MEDIAN,
        value=360,
        unit="min",
    )
    assert _canonicalize_anchor_origin(anchor).origin is AnchorOrigin.AGGREGATE


def test_shared_minute_unit_comparison_is_audited_as_two_values() -> None:
    spans = _insight_numeric_spans("Recorded awake time rose from 53 to 64 minutes.")
    assert [(item.value, item.unit) for item in spans] == [(53.0, "min"), (64.0, "min")]


def test_directional_minute_phrase_is_visible_to_anchor_validation() -> None:
    assert _anchor_value_is_visible("You spent 10 fewer minutes awake.", -10, "min")
    assert _anchor_value_is_visible("You spent 16 fewer minutes awake.", 16, "min")


def test_observation_anchors_resolve_against_profile() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    assert validate_observation(_observation(), profile, insight) == []


def test_selection_rejects_multiple_unobserved_conditions() -> None:
    library = load_generation_card_library()
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": "caffeine-close-to-sleep-may-disrupt-sleep",
                "case_relevance": "Repeated short sleep.",
                "matched_outcomes": ["total_sleep_min"],
            },
            {
                "evidence_key": "late-light-or-screen-use-may-delay-sleep",
                "case_relevance": "Repeated later sleep timing.",
                "matched_outcomes": ["sleep_onset_clock_min"],
            },
        ],
        selection_summary="Two conditional cards should not be selected together.",
    )
    issues = validate_selection(selection, _observation(), library)
    assert "more than one unobserved condition" in " ".join(issues)


def test_limited_time_in_bed_prefers_direct_duration_action_over_caffeine() -> None:
    profile = _stable_profile(sleep_min=360, time_in_bed_min=390, efficiency_pct=92)
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    direct = {card.evidence_key for card in pools.direct}
    conditional = {card.evidence_key for card in pools.conditional}
    assert "allow-more-time-for-sleep" in direct
    assert "adults-18-to-64-usually-need-at-least-7-hours" in direct
    assert "caffeine-close-to-sleep-may-disrupt-sleep" not in conditional
    assert "maintain-a-stable-adequate-sleep-pattern" not in direct


def test_recent_limited_time_in_bed_is_actionable_even_when_earlier_time_was_ample() -> None:
    observations = []
    for index in range(1, 15):
        recent = index > 7
        observations.append(
            DailyObservation(
                sleep_date=date(2026, 1, 1) + timedelta(days=index - 1),
                day_index=index,
                day_of_week=(date(2026, 1, 1) + timedelta(days=index - 1)).strftime("%A"),
                is_weekend=(date(2026, 1, 1) + timedelta(days=index - 1)).weekday() >= 5,
                metrics={
                    "total_sleep_min": 328.0 if recent else 470.0,
                    "time_in_bed_min": 412.0 if recent else 528.0,
                    "sleep_efficiency_pct": 89.8 if recent else 89.2,
                },
            )
        )
    profile = SleepProfile(
        case_id="case_recent_contraction",
        age_at_case_start_years=42,
        valid_main_sleep_nights=14,
        calendar_span_days=14,
        coverage_pct=100,
        daily_observations=observations,
        metric_summaries=[
            MetricSummary(metric_id="total_sleep_min", unit="min", n=14, median=399, mad=71),
            MetricSummary(metric_id="time_in_bed_min", unit="min", n=14, median=470, mad=58),
            MetricSummary(
                metric_id="sleep_efficiency_pct", unit="%", n=14, median=89.5, mad=0.3
            ),
        ],
    )
    keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            profile, build_longitudinal_insight(profile), load_generation_card_library()
        ).all
    }
    assert "allow-more-time-for-sleep" in keys
    assert "persistent-short-sleep-needs-clinical-context" not in keys


def test_persistent_short_sleep_with_ample_time_routes_conditional_clinical_context() -> None:
    profile = _stable_profile(sleep_min=360, time_in_bed_min=450, efficiency_pct=90)
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    direct = {card.evidence_key for card in pools.direct}
    conditional = {card.evidence_key for card in pools.conditional}
    assert "allow-more-time-for-sleep" not in direct
    assert "persistent-short-sleep-needs-clinical-context" in conditional


def test_short_sleep_with_ample_time_and_reduced_continuity_allows_caffeine() -> None:
    profile = _stable_profile(
        sleep_min=360,
        time_in_bed_min=450,
        efficiency_pct=80,
        waso_min=50,
    )
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    direct = {card.evidence_key for card in pools.direct}
    conditional = {card.evidence_key for card in pools.conditional}
    assert "allow-more-time-for-sleep" not in direct
    assert "caffeine-close-to-sleep-may-disrupt-sleep" in conditional
    assert "persistent-short-sleep-needs-clinical-context" not in conditional


def _profile_with_later_timing(
    onset_direction_min: float, wake_direction_min: float
) -> SleepProfile:
    profile = _stable_profile(sleep_min=450, time_in_bed_min=510, efficiency_pct=88)
    timing_summaries = [
        MetricSummary(
            metric_id="sleep_onset_clock_min", unit="clock_minute", n=14, median=1380, mad=30
        ),
        MetricSummary(
            metric_id="final_wake_clock_min", unit="clock_minute", n=14, median=420, mad=30
        ),
    ]
    timing_trends = [
        LongitudinalSummary(
            metric_id=metric_id,
            unit="clock_minute",
            n=14,
            span_days=14,
            sampling_density_pct=100,
            window_tier=WindowTier.ADEQUATE,
            recent_minus_earlier=direction_min,
            slope_per_7_days=direction_min / 2,
            comparison_direction_agrees_with_slope=True,
        )
        for metric_id, direction_min in (
            ("sleep_onset_clock_min", onset_direction_min),
            ("final_wake_clock_min", wake_direction_min),
        )
    ]
    return profile.model_copy(
        update={
            "metric_summaries": [*profile.metric_summaries, *timing_summaries],
            "longitudinal_summaries": timing_trends,
        }
    )


def test_whole_period_shift_routes_clinical_timing_without_late_light_competition() -> None:
    profile = _profile_with_later_timing(45, 40)
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    conditional = {card.evidence_key for card in pools.conditional}
    assert "unwanted-sleep-timing-may-need-clinical-review" in conditional
    assert "late-light-or-screen-use-may-delay-sleep" not in conditional


def test_onset_specific_later_shift_routes_late_light_without_clinical_timing() -> None:
    profile = _profile_with_later_timing(45, 0)
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    conditional = {card.evidence_key for card in pools.conditional}
    assert "late-light-or-screen-use-may-delay-sleep" in conditional
    assert "unwanted-sleep-timing-may-need-clinical-review" not in conditional


def test_router_treats_six_hours_59_minutes_as_unit_edge_not_action_target() -> None:
    profile = _stable_profile(sleep_min=419, time_in_bed_min=430, efficiency_pct=97.4)
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    keys = {card.evidence_key for card in pools.all}
    assert "adults-18-to-64-usually-need-at-least-7-hours" not in keys
    assert "allow-more-time-for-sleep" not in keys
    assert "maintain-a-stable-adequate-sleep-pattern" in keys


def test_continuity_only_does_not_randomly_route_exposure_cards() -> None:
    profile = _stable_profile(sleep_min=450, time_in_bed_min=540, efficiency_pct=83.3)
    pools = candidate_evidence_pools(
        profile, build_longitudinal_insight(profile), load_generation_card_library()
    )
    keys = {card.evidence_key for card in pools.all}
    assert "sleep-efficiency-below-85-percent-may-reflect-poor-continuity" in keys
    assert "persistent-sleep-difficulty-needs-clinical-context" in keys
    assert "caffeine-close-to-sleep-may-disrupt-sleep" not in keys
    assert "alcohol-close-to-sleep-may-disrupt-continuity" not in keys
    assert "nighttime-noise-may-disrupt-sleep" not in keys


def test_stable_waso_crossing_does_not_override_otherwise_adequate_pattern() -> None:
    profile = _stable_profile(
        sleep_min=450,
        time_in_bed_min=510,
        efficiency_pct=88.2,
        waso_min=35,
        age=67,
    )
    keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            profile,
            build_longitudinal_insight(profile),
            load_generation_card_library(),
        ).all
    }
    assert "waso-quality-reference-for-adults-65-plus" not in keys
    assert "waso-quality-reference-for-adults-18-to-64" not in keys
    assert "maintain-a-stable-adequate-sleep-pattern" in keys


def test_router_uses_age_specific_waso_reference_when_efficiency_is_also_low() -> None:
    profile = _stable_profile(
        sleep_min=420,
        time_in_bed_min=510,
        efficiency_pct=82.4,
        waso_min=35,
        age=67,
    )
    keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            profile,
            build_longitudinal_insight(profile),
            load_generation_card_library(),
        ).all
    }
    assert "waso-quality-reference-for-adults-65-plus" in keys
    assert "waso-quality-reference-for-adults-18-to-64" not in keys


def test_router_uses_sleep_latency_only_when_repeated_center_exceeds_reference() -> None:
    high = _stable_profile(
        sleep_min=450,
        time_in_bed_min=510,
        efficiency_pct=88.2,
        latency_min=42,
    )
    high_keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            high,
            build_longitudinal_insight(high),
            load_generation_card_library(),
        ).all
    }
    assert "sleep-latency-quality-reference-for-adults" in high_keys

    zeros = _stable_profile(
        sleep_min=450,
        time_in_bed_min=510,
        efficiency_pct=88.2,
        latency_min=0,
    )
    zero_keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            zeros,
            build_longitudinal_insight(zeros),
            load_generation_card_library(),
        ).all
    }
    assert "sleep-latency-quality-reference-for-adults" not in zero_keys


def test_midpoint_routes_coherent_timing_shift_when_components_are_noisy() -> None:
    profile = _stable_profile(
        sleep_min=450,
        time_in_bed_min=510,
        efficiency_pct=88.2,
    )
    profile = profile.model_copy(
        update={
            "metric_summaries": [
                *profile.metric_summaries,
                MetricSummary(
                    metric_id="sleep_onset_clock_min",
                    unit="clock_minute",
                    n=14,
                    median=1380,
                    mad=50,
                ),
                MetricSummary(
                    metric_id="final_wake_clock_min",
                    unit="clock_minute",
                    n=14,
                    median=420,
                    mad=100,
                ),
                MetricSummary(
                    metric_id="sleep_midpoint_clock_min",
                    unit="clock_minute",
                    n=14,
                    median=180,
                    mad=20,
                ),
            ],
            "weekday_weekend_comparisons": [
                WeekpartComparison(
                    metric_id="sleep_onset_clock_min",
                    unit="clock_minute",
                    weekday_n=10,
                    weekend_n=4,
                    weekday_center=1375,
                    weekend_center=1385,
                    weekend_minus_weekday=10,
                    reportable=True,
                ),
                WeekpartComparison(
                    metric_id="final_wake_clock_min",
                    unit="clock_minute",
                    weekday_n=10,
                    weekend_n=4,
                    weekday_center=375,
                    weekend_center=465,
                    weekend_minus_weekday=90,
                    reportable=True,
                ),
                WeekpartComparison(
                    metric_id="sleep_midpoint_clock_min",
                    unit="clock_minute",
                    weekday_n=10,
                    weekend_n=4,
                    weekday_center=155,
                    weekend_center=205,
                    weekend_minus_weekday=50,
                    reportable=True,
                ),
            ],
        }
    )
    keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            profile,
            build_longitudinal_insight(profile),
            load_generation_card_library(),
        ).all
    }
    assert "make-sleep-timing-more-consistent" in keys
    assert "sleep-timing-alone-does-not-confirm-circadian-phase" in keys
    assert generation_case_signals(profile.model_dump(mode="json")) == {
        "weekday_weekend_whole_sleep_period_shift_is_midpoint_decisive": True
    }


def test_borderline_midpoint_difference_does_not_trigger_timing_guidance() -> None:
    profile = _stable_profile(
        sleep_min=484,
        time_in_bed_min=540,
        efficiency_pct=89.6,
    )
    profile = profile.model_copy(
        update={
            "metric_summaries": [
                *profile.metric_summaries,
                MetricSummary(
                    metric_id="sleep_onset_clock_min",
                    unit="clock_minute",
                    n=29,
                    median=1278.5,
                    mad=11,
                ),
                MetricSummary(
                    metric_id="final_wake_clock_min",
                    unit="clock_minute",
                    n=29,
                    median=378.5,
                    mad=26,
                ),
                MetricSummary(
                    metric_id="sleep_midpoint_clock_min",
                    unit="clock_minute",
                    n=29,
                    median=107,
                    mad=14.75,
                ),
            ],
            "weekday_weekend_comparisons": [
                WeekpartComparison(
                    metric_id="sleep_onset_clock_min",
                    unit="clock_minute",
                    weekday_n=20,
                    weekend_n=9,
                    weekday_center=1275.75,
                    weekend_center=1283.5,
                    weekend_minus_weekday=7.75,
                    reportable=True,
                ),
                WeekpartComparison(
                    metric_id="final_wake_clock_min",
                    unit="clock_minute",
                    weekday_n=20,
                    weekend_n=9,
                    weekday_center=376.25,
                    weekend_center=381.5,
                    weekend_minus_weekday=5.25,
                    reportable=True,
                ),
                WeekpartComparison(
                    metric_id="sleep_midpoint_clock_min",
                    unit="clock_minute",
                    weekday_n=20,
                    weekend_n=9,
                    weekday_center=105.625,
                    weekend_center=120.5,
                    weekend_minus_weekday=14.875,
                    reportable=True,
                ),
            ],
        }
    )
    keys = {
        card.evidence_key
        for card in candidate_evidence_pools(
            profile,
            build_longitudinal_insight(profile),
            load_generation_card_library(),
        ).all
    }
    assert "make-sleep-timing-more-consistent" not in keys
    assert generation_case_signals(profile.model_dump(mode="json")) == {
        "weekday_weekend_whole_sleep_period_shift_is_midpoint_decisive": False
    }


def test_final_used_evidence_must_equal_minimal_selection() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": "allow-more-time-for-sleep",
                "case_relevance": "Sleep and time in bed were both repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was typically about 6 hours.",
        section_2="Sleep and time in bed were both repeatedly short.",
        section_3="When practical, start the sleep period a little earlier.",
        section_4="Over the next two weeks, notice whether sleep lasts longer than 6 hours.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=SectionGrounding(),
        section_3_grounding=SectionGrounding(),
        section_4_grounding=SectionGrounding(outcome_metrics=["total_sleep_min"]),
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "Final used Evidence must equal" in " ".join(issues)


def test_recommendation_rejects_an_invented_action_title() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep and time in bed were both repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was typically about 6 hours.",
        section_2="Sleep and time in bed were both repeatedly short.",
        section_3=(
            "**Completely invented heading**\n"
            "Your repeated short sleep makes additional sleep time relevant. "
            "When practical, leave more time for sleep."
        ),
        section_4="Over the next several nights, see whether sleep lasts longer.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )

    issues = validate_recommendation(draft, profile, insight, selection, library)
    joined = " ".join(issues)
    assert "outside allowed_action_titles" in joined
    assert "exactly one controlled title for 'allow-more-time-for-sleep'" in joined


def test_conditional_clinical_discussion_does_not_require_treatment_outcome() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "persistent-sleep-difficulty-needs-clinical-context"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "A repeated continuity concern remains unresolved.",
                "matched_outcomes": ["waso_min"],
            }
        ],
        selection_summary="Clinical context is needed only if the recorded concern is symptomatic.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep continuity varied across the record.",
        section_2="The sleep record alone cannot show whether this is a sleep disorder.",
        section_3=(
            "If you also have ongoing trouble sleeping, unusual sleepiness during the day, "
            "or trouble concentrating, discuss the pattern with a clinician."
        ),
        section_4=(
            "Before that discussion, note which of those concerns you notice and how often."
        ),
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=grounding,
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "Section 4 must name at least one grounded evaluation outcome" not in issues

    leaked = draft.model_copy(
        update={
            "section_2": (
                "The internal label persistent-sleep-difficulty-needs-clinical-context "
                "must not be shown to the reader."
            )
        }
    )
    leaked_issues = validate_recommendation(leaked, profile, insight, selection, library)
    assert "Reader text exposes internal Evidence keys" in " ".join(leaked_issues)


def test_melatonin_is_visible_only_for_selected_timing_clinical_boundary() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "unwanted-sleep-timing-may-need-clinical-review"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "The whole sleep period repeatedly shifted later.",
                "matched_outcomes": ["sleep_onset_clock_min"],
            }
        ],
        selection_summary="Clinical review remains conditional on unwanted timing.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="The whole sleep period shifted later across recent nights.",
        section_2="Whether the later timing matters depends on whether it conflicts with your needs.",
        section_3=(
            "If the timing is unwanted and affects necessary responsibilities, discuss it "
            "with a sleep specialist rather than starting or timing melatonin from these "
            "sleep records alone."
        ),
        section_4="Before the discussion, note whether the later timing continues.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["sleep_onset_clock_min"]
        ),
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "must name melatonin" not in " ".join(issues)
    missing = draft.model_copy(
        update={
            "section_3": (
                "If the timing is unwanted and affects necessary responsibilities, discuss "
                "it with a sleep specialist."
            )
        }
    )
    missing_issues = validate_recommendation(missing, profile, insight, selection, library)
    assert "must name melatonin" in " ".join(missing_issues)
    repeated = draft.model_copy(
        update={
            "section_4": (
                "If the later timing continues, discuss it with a sleep specialist and do "
                "not start melatonin from these records alone."
            )
        }
    )
    repeated_issues = validate_recommendation(
        repeated, profile, insight, selection, library
    )
    assert "repeats the melatonin guardrail" in " ".join(repeated_issues)
    unsupported_symptom = draft.model_copy(
        update={
            "section_4": (
                "If the later timing continues or affects concentration, discuss it with "
                "a sleep specialist."
            )
        }
    )
    symptom_issues = validate_recommendation(
        unsupported_symptom, profile, insight, selection, library
    )
    assert "do not add unrecorded energy" in " ".join(symptom_issues)


def test_behavior_guidance_cannot_add_generic_daytime_problem() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep and time in bed were both repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was typically about 6 hours.",
        section_2="Sleep and time in bed were both repeatedly short.",
        section_3="When practical, start the sleep period a little earlier.",
        section_4="Notice whether sleep lasts longer and whether daytime problems improve.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "daytime symptoms without selected clinical-boundary Evidence" in " ".join(issues)


def test_reader_text_rejects_threshold_first_reference_jargon() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep and time in bed were both repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was typically short.",
        section_2=(
            "Estimated awake time was above the older-adult sleep-quality reference "
            "of 30 minutes or less."
        ),
        section_3="When practical, leave more time for sleep.",
        section_4="Over the next several nights, see whether sleep lasts longer.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "reader-facing reference jargon" in " ".join(issues)


def test_reader_text_rejects_estimated_waso_wording() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep and time in bed were both repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was short, with an estimated 40 minutes awake after falling asleep.",
        section_2="Sleep and time in bed were both repeatedly short.",
        section_3=(
            "**Make more time for sleep**\n"
            "When practical, leave more time for sleep."
        ),
        section_4="Over the next several nights, see whether sleep lasts longer.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )

    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "qualified WASO wording" in " ".join(issues)


def test_follow_up_rejects_unsupported_sleep_time_withdrawal() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep and time in bed were both repeatedly short.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was repeatedly short.",
        section_2="Sleep and time in bed decreased together.",
        section_3="**Make more time for sleep**\nWhen practical, leave more time for sleep.",
        section_4=(
            "If sleep stays about the same, stop adding more time in bed for this purpose."
        ),
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )

    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "unsupported sleep-time withdrawal" in " ".join(issues)


def test_follow_up_rejects_abstract_strategy_verdict() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "caffeine-close-to-sleep-may-disrupt-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Short and interrupted sleep occurred repeatedly.",
                "matched_outcomes": ["total_sleep_min", "waso_min"],
            }
        ],
        selection_summary="Caffeine timing is one relevant unobserved factor to ask about.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was repeatedly short and interrupted.",
        section_2="Time awake after falling asleep remained elevated.",
        section_3=(
            "**Move caffeine earlier if it is near bedtime**\n"
            "If you use caffeine near bedtime, have it earlier."
        ),
        section_4=(
            "If awake time does not improve, stop treating caffeine timing as a sleep strategy."
        ),
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["waso_min"]
        ),
    )

    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "abstract strategy verdict" in " ".join(issues)


def test_analysis_does_not_repeat_an_unobserved_caffeine_condition() -> None:
    profile = _profile()
    insight = build_longitudinal_insight(profile)
    library = load_generation_card_library()
    key = "caffeine-close-to-sleep-may-disrupt-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Short and interrupted sleep occurred repeatedly.",
                "matched_outcomes": ["total_sleep_min", "waso_min"],
            }
        ],
        selection_summary="Caffeine timing is one relevant unobserved factor to ask about.",
    )
    grounding = SectionGrounding(evidence_keys=[key])
    draft = RecommendationDraft(
        case_id="case_stage1",
        section_1="Sleep was repeatedly short and interrupted.",
        section_2="Caffeine near bedtime could contribute to this pattern.",
        section_3="If you use caffeine near bedtime, have it earlier.",
        section_4="Over the next several nights, see whether sleep lasts longer.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=grounding,
        section_3_grounding=grounding,
        section_4_grounding=SectionGrounding(
            evidence_keys=[key], outcome_metrics=["total_sleep_min"]
        ),
    )
    issues = validate_recommendation(draft, profile, insight, selection, library)
    assert "Section 2 repeats an unobserved conditional factor" in " ".join(issues)


def test_caffeine_guidance_rejects_adequate_recent_sleep() -> None:
    profile = _stable_profile(
        sleep_min=480,
        time_in_bed_min=540,
        efficiency_pct=88.9,
        waso_min=60,
    )
    insight = build_longitudinal_insight(profile)
    key = "caffeine-close-to-sleep-may-disrupt-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep remained long despite repeated wakefulness.",
                "matched_outcomes": ["total_sleep_min", "waso_min"],
            }
        ],
        selection_summary="Conditional caffeine guidance.",
    )
    issues = validate_selection(
        selection,
        _observation(),
        load_generation_card_library(),
        insight=insight,
    )
    assert "requires repeated recent sleep clearly below 7 hours" in " ".join(issues)


def test_allow_more_time_rejects_ample_recent_time_in_bed() -> None:
    profile = _stable_profile(
        sleep_min=360,
        time_in_bed_min=480,
        efficiency_pct=75.0,
        waso_min=120,
    )
    insight = build_longitudinal_insight(profile)
    key = "allow-more-time-for-sleep"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep was short despite ample time in bed.",
                "matched_outcomes": ["total_sleep_min", "time_in_bed_min"],
            }
        ],
        selection_summary="Allow more time for sleep.",
    )
    issues = validate_selection(
        selection,
        _observation(),
        load_generation_card_library(),
        profile=profile,
        insight=insight,
    )
    assert "requires short sleep and less than 7 hours in bed" in " ".join(issues)


def test_activity_guidance_rejects_missing_case_association() -> None:
    profile = _stable_profile(
        sleep_min=420,
        time_in_bed_min=500,
        efficiency_pct=84.0,
        waso_min=80,
    )
    key = "regular-daytime-movement-may-support-sleep-efficiency"
    selection = EvidenceSelection(
        case_id="case_stage1",
        selected=[
            {
                "evidence_key": key,
                "case_relevance": "Sleep efficiency was low.",
                "matched_outcomes": ["sleep_efficiency_pct"],
            }
        ],
        selection_summary="Use regular daytime movement.",
    )
    issues = validate_selection(
        selection,
        _observation(),
        load_generation_card_library(),
        profile=profile,
        insight=build_longitudinal_insight(profile),
    )
    assert "requires a reportable positive within-person" in " ".join(issues)

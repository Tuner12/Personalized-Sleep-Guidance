"""Minimal audited contract for a joint four-section recommendation."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from sleepguide.core.contracts import ContractModel, LongitudinalInsight, SleepProfile
from sleepguide.core.evidence import CONDITIONAL_GUIDANCE_DOSSIER_IDS, ReadableEvidenceBank

_HISTORICAL_SUCCESS_TARGET = re.compile(
    r"(?:\b(?:toward|towards|reach|reaching|return(?:ing)?\s+to|back\s+to)\b"
    r"[^.!?]{0,120}\b(?:earlier|previous|prior|historical)\b|"
    r"\b(?:earlier|previous|prior|historical)\b[^.!?]{0,120}"
    r"\b(?:target|goal|success|successful|helping)\b)",
    re.I,
)
_NEGATED_HISTORICAL_TARGET = re.compile(
    r"\b(?:rather\s+than|not|avoid|without)\b[^.!?]{0,180}"
    r"\b(?:earlier|previous|prior|historical)\b[^.!?]{0,120}"
    r"\b(?:target|goal|success|successful|helping)\b",
    re.I,
)

_CURRENT_BASELINE_STATISTICS = {
    "mean",
    "median",
    "recent_7_center",
    "recent_window_center",
}


class AnchorOrigin(StrEnum):
    DAILY_OBSERVATION = "daily_observation"
    AGGREGATE = "aggregate"
    LONGITUDINAL = "longitudinal"
    WEEKPART = "weekpart"
    ASSOCIATION = "association"
    INSIGHT = "insight"


class AnchorStatistic(StrEnum):
    VALUE = "value"
    N = "n"
    MEAN = "mean"
    MEDIAN = "median"
    SD = "sd"
    MAD = "mad"
    IQR = "iqr"
    P10 = "p10"
    P90 = "p90"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    FIRST_VALUE = "first_value"
    LAST_VALUE = "last_value"
    SPAN_DAYS = "span_days"
    SAMPLING_DENSITY_PCT = "sampling_density_pct"
    EARLIER_WINDOW_CENTER = "earlier_window_center"
    EARLIER_WINDOW_MAD = "earlier_window_mad"
    EARLIER_WINDOW_IQR = "earlier_window_iqr"
    EARLIER_WINDOW_N = "earlier_window_n"
    EARLIER_WINDOW_CALENDAR_SPAN_DAYS = "earlier_window_calendar_span_days"
    RECENT_WINDOW_CENTER = "recent_window_center"
    RECENT_WINDOW_MAD = "recent_window_mad"
    RECENT_WINDOW_IQR = "recent_window_iqr"
    RECENT_WINDOW_N = "recent_window_n"
    RECENT_WINDOW_CALENDAR_SPAN_DAYS = "recent_window_calendar_span_days"
    RECENT_MINUS_EARLIER = "recent_minus_earlier"
    RECENT_MINUS_EARLIER_MAD = "recent_minus_earlier_mad"
    SLOPE_PER_7_DAYS = "slope_per_7_days"
    WEEKDAY_N = "weekday_n"
    WEEKEND_N = "weekend_n"
    WEEKDAY_CENTER = "weekday_center"
    WEEKEND_CENTER = "weekend_center"
    WEEKEND_MINUS_WEEKDAY = "weekend_minus_weekday"
    PAIRED_NIGHTS = "paired_nights"
    SPEARMAN_RHO = "spearman_rho"
    HIGH_MINUS_LOW_SLEEP = "high_minus_low_sleep"
    RECENT_7_CENTER = "recent_7_center"
    PREVIOUS_7_CENTER = "previous_7_center"
    EARLIER_HISTORY_CENTER = "earlier_history_center"
    RECENT_MINUS_PREVIOUS_7 = "recent_minus_previous_7"
    RECENT_MINUS_EARLIER_HISTORY = "recent_minus_earlier_history"


_CLOCK_DURATION_STATISTICS = {
    AnchorStatistic.SD,
    AnchorStatistic.MAD,
    AnchorStatistic.IQR,
    AnchorStatistic.EARLIER_WINDOW_MAD,
    AnchorStatistic.EARLIER_WINDOW_IQR,
    AnchorStatistic.RECENT_WINDOW_MAD,
    AnchorStatistic.RECENT_WINDOW_IQR,
    AnchorStatistic.RECENT_MINUS_EARLIER,
    AnchorStatistic.RECENT_MINUS_EARLIER_MAD,
    AnchorStatistic.SLOPE_PER_7_DAYS,
    AnchorStatistic.WEEKEND_MINUS_WEEKDAY,
    AnchorStatistic.HIGH_MINUS_LOW_SLEEP,
}


class CaseMetricAnchor(ContractModel):
    metric_id: str
    origin: AnchorOrigin
    statistic: AnchorStatistic
    value: float
    unit: str
    related_metric_id: str | None = None
    observation_day_index: int | None = Field(default=None, ge=1)

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_count_unit(cls, value: object) -> object:
        if isinstance(value, str):
            aliases = {
                "nights": "night",
                "days": "day",
                "unitless": "rho",
                "coefficient": "rho",
                "correlation": "rho",
                "percentage point": "%",
                "percentage points": "%",
            }
            return aliases.get(value.strip().lower(), value)
        return value

    @model_validator(mode="after")
    def validate_locator_shape(self) -> CaseMetricAnchor:
        if self.unit == "clock_minute" and self.statistic in _CLOCK_DURATION_STATISTICS:
            self.unit = "min"
        if self.origin is AnchorOrigin.ASSOCIATION:
            if not self.related_metric_id:
                raise ValueError("association anchor requires related_metric_id")
        elif self.related_metric_id is not None:
            raise ValueError("related_metric_id is allowed only for association anchors")
        if self.origin is AnchorOrigin.DAILY_OBSERVATION:
            if self.observation_day_index is None:
                raise ValueError("daily-observation anchor requires observation_day_index")
        elif self.observation_day_index is not None:
            raise ValueError("observation_day_index is allowed only for daily-observation anchors")
        return self


class EvidenceReferenceAnchor(ContractModel):
    dossier_id: str
    metric_id: str
    comparator: str
    value: float
    unit: str
    label: str


class SectionGrounding(ContractModel):
    """Audit metadata for one complete reader-visible Section."""

    evidence_dossier_ids: list[str] = Field(min_length=1)
    case_anchors: list[CaseMetricAnchor] = Field(default_factory=list)
    evidence_reference_anchors: list[EvidenceReferenceAnchor] = Field(default_factory=list)


class ReaderRecommendation(ContractModel):
    section_1: str = Field(min_length=1)
    section_2: str = Field(min_length=1)
    section_3: str = Field(min_length=1)
    section_4: str = Field(min_length=1)


class RecommendationDraft(ContractModel):
    """Four reader Sections plus compact, Section-level grounding metadata."""

    case_id: str
    section_1: str = Field(min_length=1)
    section_2: str = Field(min_length=1)
    section_3: str = Field(min_length=1)
    section_4: str = Field(min_length=1)
    section_2_grounding: SectionGrounding
    section_3_grounding: SectionGrounding
    section_4_grounding: SectionGrounding

    @model_validator(mode="after")
    def validate_reader_text(self) -> RecommendationDraft:
        visible = "\n".join(self.render_sections().values())
        forbidden = {
            "aggregated observations",
            "vendor-classified",
            "device-classified",
            "mandatory safety",
            "markedly short",
            "recorded median",
            "selected evidence",
            "selected dossier",
            "evidence dossier",
            "counterbalance",
            "decision-relevant",
            "pre-trial baseline",
            "rolling",
            "provisional",
            "multi-night",
            "trajectory",
        }
        present = sorted(term for term in forbidden if term in visible.casefold())
        jargon = {
            "unsupported source label": (
                r"\b(?:trackers?|wearables?|fitbit|devices?|sensors?|wrist[- ]based|"
                r"wrist devices?|consumer sleep records?|consumer[- ]recorded|"
                r"technology[- ]derived|algorithm[- ]estimated)\b"
            ),
            "Spearman/rho": r"\b(?:spearman|rho|correlation coefficient)\b",
            "co-varied": r"\bco-?varied\b",
            "paired nights": r"\bpaired nights\b",
            "reportable": r"\breportable\b",
            "window-center terminology": r"\b(?:recent[- ]window center|whole[- ]record median)\b",
            "dispersion abbreviation": r"\b(?:MAD|IQR)\b",
            "internal period label": r"\bblocks?\b",
            "internal center label": r"\bcenter(?:ed)?\b",
            "internal baseline label": r"\bbaseline\b",
        }
        present.extend(
            label for label, pattern in jargon.items() if re.search(pattern, visible, re.I)
        )
        if present:
            raise ValueError(f"reader recommendation exposes internal language: {sorted(present)}")
        return self

    def render_sections(self) -> dict[str, str]:
        return {
            "section_1": self.section_1.strip(),
            "section_2": self.section_2.strip(),
            "section_3": self.section_3.strip(),
            "section_4": self.section_4.strip(),
        }

    def to_reader_recommendation(self) -> ReaderRecommendation:
        return ReaderRecommendation.model_validate(self.render_sections())

    def grounding_for(self, section: str) -> SectionGrounding | None:
        return {
            "section_1": None,
            "section_2": self.section_2_grounding,
            "section_3": self.section_3_grounding,
            "section_4": self.section_4_grounding,
        }[section]


_AGGREGATE_STATISTICS = {
    AnchorStatistic.N,
    AnchorStatistic.MEAN,
    AnchorStatistic.MEDIAN,
    AnchorStatistic.SD,
    AnchorStatistic.MAD,
    AnchorStatistic.IQR,
    AnchorStatistic.P10,
    AnchorStatistic.P90,
    AnchorStatistic.MINIMUM,
    AnchorStatistic.MAXIMUM,
    AnchorStatistic.FIRST_VALUE,
    AnchorStatistic.LAST_VALUE,
}
_LONGITUDINAL_FIELDS = {
    AnchorStatistic.N: (None, "n", "night"),
    AnchorStatistic.SPAN_DAYS: (None, "span_days", "day"),
    AnchorStatistic.SAMPLING_DENSITY_PCT: (None, "sampling_density_pct", "%"),
    AnchorStatistic.RECENT_MINUS_EARLIER: (None, "recent_minus_earlier", None),
    AnchorStatistic.RECENT_MINUS_EARLIER_MAD: (None, "recent_minus_earlier_mad", None),
    AnchorStatistic.SLOPE_PER_7_DAYS: (None, "slope_per_7_days", None),
    AnchorStatistic.EARLIER_WINDOW_CENTER: ("earlier_window", "center", None),
    AnchorStatistic.EARLIER_WINDOW_MAD: ("earlier_window", "mad", None),
    AnchorStatistic.EARLIER_WINDOW_IQR: ("earlier_window", "iqr", None),
    AnchorStatistic.EARLIER_WINDOW_N: ("earlier_window", "n", "night"),
    AnchorStatistic.EARLIER_WINDOW_CALENDAR_SPAN_DAYS: (
        "earlier_window",
        "calendar_span_days",
        "day",
    ),
    AnchorStatistic.RECENT_WINDOW_CENTER: ("recent_window", "center", None),
    AnchorStatistic.RECENT_WINDOW_MAD: ("recent_window", "mad", None),
    AnchorStatistic.RECENT_WINDOW_IQR: ("recent_window", "iqr", None),
    AnchorStatistic.RECENT_WINDOW_N: ("recent_window", "n", "night"),
    AnchorStatistic.RECENT_WINDOW_CALENDAR_SPAN_DAYS: (
        "recent_window",
        "calendar_span_days",
        "day",
    ),
}
_WEEKPART_FIELDS = {
    AnchorStatistic.WEEKDAY_N: ("weekday_n", "night"),
    AnchorStatistic.WEEKEND_N: ("weekend_n", "night"),
    AnchorStatistic.WEEKDAY_CENTER: ("weekday_center", None),
    AnchorStatistic.WEEKEND_CENTER: ("weekend_center", None),
    AnchorStatistic.WEEKEND_MINUS_WEEKDAY: ("weekend_minus_weekday", None),
}
_ASSOCIATION_FIELDS = {
    AnchorStatistic.PAIRED_NIGHTS: ("paired_nights", "night"),
    AnchorStatistic.SPEARMAN_RHO: ("spearman_rho", "rho"),
    AnchorStatistic.HIGH_MINUS_LOW_SLEEP: ("high_minus_low_sleep", "sleep_unit"),
}


def _anchor_value_is_visible(text: str, value: float, unit: str) -> bool:
    display = abs(value)
    rounded = int(display + 0.5)
    if unit in {"min", "minute", "minutes"}:
        # Accept either an exact decimal supplied by the deterministic statistic or
        # ordinary whole-minute display rounding. Half-minute medians are common when
        # an even number of nights is summarized.
        minute_tokens = {f"{display:g}", str(rounded)}
        if any(
            re.search(
                rf"\b{re.escape(token)}(?:\.0+)?[\s-]*(?:m|min|mins|minute|minutes)\b",
                text,
                re.I,
            )
            for token in minute_tokens
        ):
            return True
        if re.search(
            rf"\b{rounded}(?:\.0+)?\s*(?:to|versus|vs\.?|and)\s*"
            rf"\d+(?:\.\d+)?\s*(?:m|min|mins|minute|minutes)\b",
            text,
            re.I,
        ):
            return True
        hours, minutes = divmod(rounded, 60)
        if re.search(
            rf"\bat\s+least\s+{hours}(?:\.0+)?\s*"
            rf"(?:h|hr|hrs|hour|hours)\b|"
            rf"\b{hours}(?:\.0+)?\s+or\s+more\s*"
            rf"(?:h|hr|hrs|hour|hours)\b",
            text,
            re.I,
        ):
            return True
        hours_seen = re.search(
            rf"\b{hours}[\s-]*(?:h|hr|hrs|hour|hours)\b", text, re.I
        )
        minutes_seen = re.search(
            rf"\b{minutes}[\s-]*(?:m|min|mins|minute|minutes)\b", text, re.I
        )
        if hours_seen and (minutes == 0 or minutes_seen):
            return True
        # Directional comparisons commonly place the modifier between the value and
        # its unit, for example "10 fewer minutes".  This is still an exact value.
        return bool(
            re.search(
                rf"\b{rounded}(?:\.0+)?\s+(?:fewer|more|less|additional|extra)\s+"
                rf"(?:m|min|mins|minute|minutes)\b",
                text,
                re.I,
            )
        )
    if unit == "clock_minute":
        total_seconds = int(round(display * 60)) % (24 * 60 * 60)
        second_hour, remainder = divmod(total_seconds, 3600)
        second_minute, second = divmod(remainder, 60)
        if second and re.search(
            rf"\b{second_hour % 12 or 12}:{second_minute:02d}:{second:02d}\s*"
            rf"{'a' if second_hour < 12 else 'p'}\.?\s*m\.?\b",
            text,
            re.I,
        ):
            return True
        # A median clock position may land at :30 seconds. Reader text displays only
        # minutes, so either adjacent minute is a faithful rounding of that value.
        clock_candidates = {int(display), rounded}
        for candidate in clock_candidates:
            minute_of_day = candidate % 1440
            hour, minute = divmod(minute_of_day, 60)
            meridiem = "a" if hour < 12 else "p"
            if re.search(
                rf"\b{hour % 12 or 12}:{minute:02d}\s*{meridiem}\.?\s*m\.?\b",
                text,
                re.I,
            ):
                return True
        return False
    numbers = {
        float(token.replace(",", "")) for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)
    }
    # Percentages are displayed to one decimal place.  Include both neighboring
    # one-decimal renderings at an exact half-tie (for example 88.05 -> 88.0 or
    # 88.1), since different renderers use different tie-breaking conventions.
    candidates = {
        value,
        display,
        float(rounded),
        round(value, 1),
        round(value + 1e-8, 1),
        round(value - 1e-8, 1),
        round(value, 2),
    }
    return any(abs(number - candidate) <= 1e-6 for number in numbers for candidate in candidates)


def resolve_profile_anchor(
    profile: SleepProfile, anchor: CaseMetricAnchor
) -> tuple[float, str] | None:
    if anchor.origin is AnchorOrigin.DAILY_OBSERVATION:
        if anchor.statistic is not AnchorStatistic.VALUE:
            return None
        observation = next(
            (
                item
                for item in profile.daily_observations
                if item.day_index == anchor.observation_day_index
            ),
            None,
        )
        summary = next(
            (item for item in profile.metric_summaries if item.metric_id == anchor.metric_id), None
        )
        value = observation.metrics.get(anchor.metric_id) if observation is not None else None
        return (float(value), summary.unit) if value is not None and summary is not None else None
    if anchor.origin is AnchorOrigin.AGGREGATE:
        if anchor.statistic not in _AGGREGATE_STATISTICS:
            return None
        summary = next(
            (item for item in profile.metric_summaries if item.metric_id == anchor.metric_id), None
        )
        if summary is None:
            return None
        value = getattr(summary, anchor.statistic.value)
        unit = "night" if anchor.statistic is AnchorStatistic.N else summary.unit
        if unit == "clock_minute" and anchor.statistic in _CLOCK_DURATION_STATISTICS:
            unit = "min"
        return (float(value), unit) if value is not None else None
    if anchor.origin is AnchorOrigin.LONGITUDINAL:
        spec = _LONGITUDINAL_FIELDS.get(anchor.statistic)
        summary = next(
            (item for item in profile.longitudinal_summaries if item.metric_id == anchor.metric_id),
            None,
        )
        if spec is None or summary is None:
            return None
        window_name, field_name, fixed_unit = spec
        source = getattr(summary, window_name) if window_name else summary
        value = getattr(source, field_name) if source is not None else None
        unit = fixed_unit or summary.unit
        if unit == "clock_minute" and anchor.statistic in _CLOCK_DURATION_STATISTICS:
            unit = "min"
        return (float(value), unit) if value is not None else None
    if anchor.origin is AnchorOrigin.WEEKPART:
        spec = _WEEKPART_FIELDS.get(anchor.statistic)
        comparison = next(
            (
                item
                for item in profile.weekday_weekend_comparisons
                if item.metric_id == anchor.metric_id and item.reportable
            ),
            None,
        )
        if spec is None or comparison is None:
            return None
        field_name, fixed_unit = spec
        value = getattr(comparison, field_name)
        unit = fixed_unit or comparison.unit
        if unit == "clock_minute" and anchor.statistic in _CLOCK_DURATION_STATISTICS:
            unit = "min"
        return (float(value), unit) if value is not None else None
    if anchor.origin is AnchorOrigin.ASSOCIATION:
        spec = _ASSOCIATION_FIELDS.get(anchor.statistic)
        association = next(
            (
                item
                for item in profile.daily_context_associations
                if item.reportable
                and item.context_metric_id == anchor.metric_id
                and item.sleep_metric_id == anchor.related_metric_id
            ),
            None,
        )
        if spec is None or association is None:
            return None
        field_name, unit_role = spec
        value = getattr(association, field_name)
        unit = association.unit_sleep if unit_role == "sleep_unit" else unit_role
        return (float(value), unit) if value is not None and unit is not None else None
    return None


def resolve_insight_anchor(
    insight: LongitudinalInsight | None, anchor: CaseMetricAnchor
) -> tuple[float, str] | None:
    if insight is None or anchor.origin is not AnchorOrigin.INSIGHT:
        return None
    item = next(
        (
            entry
            for entry in insight.recent_7_night_comparison
            if entry.metric_id == anchor.metric_id
        ),
        None,
    )
    if item is None:
        return None
    fields = {
        AnchorStatistic.RECENT_7_CENTER: (
            item.recent_7_nights.center if item.recent_7_nights else None
        ),
        AnchorStatistic.PREVIOUS_7_CENTER: (
            item.previous_7_nights.center if item.previous_7_nights else None
        ),
        AnchorStatistic.EARLIER_HISTORY_CENTER: (
            item.earlier_history.center if item.earlier_history else None
        ),
        AnchorStatistic.RECENT_MINUS_PREVIOUS_7: item.recent_minus_previous,
        AnchorStatistic.RECENT_MINUS_EARLIER_HISTORY: item.recent_minus_earlier_history,
    }
    value = fields.get(anchor.statistic)
    unit = item.unit
    if unit == "clock_minute" and anchor.statistic in {
        AnchorStatistic.RECENT_MINUS_PREVIOUS_7,
        AnchorStatistic.RECENT_MINUS_EARLIER_HISTORY,
    }:
        unit = "min"
    return (float(value), unit) if value is not None else None


def _validate_grounding(
    section: str,
    text: str,
    grounding: SectionGrounding,
    profile: SleepProfile,
    insight: LongitudinalInsight | None,
    bank: ReadableEvidenceBank,
) -> list[str]:
    issues: list[str] = []
    by_id = {item.dossier_id: item for item in bank.dossiers}
    if len(grounding.evidence_dossier_ids) != len(set(grounding.evidence_dossier_ids)):
        issues.append(f"{section}: duplicate evidence dossier IDs")
    for dossier_id in grounding.evidence_dossier_ids:
        dossier = by_id.get(dossier_id)
        if dossier is None or not dossier.runtime_enabled:
            issues.append(f"{section}: unknown or disabled dossier {dossier_id!r}")
    for anchor in grounding.case_anchors:
        resolved = (
            resolve_insight_anchor(insight, anchor)
            if anchor.origin is AnchorOrigin.INSIGHT
            else resolve_profile_anchor(profile, anchor)
        )
        if resolved is None:
            issues.append(
                f"{section}: unresolved case anchor "
                f"{anchor.origin.value}/{anchor.metric_id}/{anchor.statistic.value}"
            )
            continue
        expected_value, expected_unit = resolved
        if anchor.unit != expected_unit or abs(anchor.value - expected_value) > 1e-6:
            issues.append(f"{section}: case anchor differs from canonical Profile value")
        elif not _anchor_value_is_visible(text, expected_value, expected_unit):
            issues.append(
                f"{section}: case anchor {anchor.origin.value}/{anchor.metric_id}/"
                f"{anchor.statistic.value}={expected_value:g} {expected_unit} is not "
                "represented in reader text; remove the unused anchor or state its "
                "rounded value"
            )
    for anchor in grounding.evidence_reference_anchors:
        if anchor.dossier_id not in grounding.evidence_dossier_ids:
            issues.append(f"{section}: reference dossier is not cited")
            continue
        dossier = by_id.get(anchor.dossier_id)
        declared = dossier is not None and any(
            value.metric_id == anchor.metric_id
            and value.comparator == anchor.comparator
            and value.unit == anchor.unit
            and value.label == anchor.label
            and abs(value.value - anchor.value) <= 1e-9
            for value in dossier.reference_values
        )
        if not declared:
            issues.append(f"{section}: undeclared evidence reference value")
        elif not _anchor_value_is_visible(text, anchor.value, anchor.unit):
            issues.append(f"{section}: evidence reference is not represented in reader text")
    return issues


def validate_recommendation(
    draft: RecommendationDraft,
    profile: SleepProfile,
    bank: ReadableEvidenceBank,
    *,
    insight: LongitudinalInsight | None = None,
) -> list[str]:
    """Validate Section-level Evidence links and every declared numeric anchor."""

    issues: list[str] = []
    for section in ("section_2", "section_3", "section_4"):
        issues.extend(
            _validate_grounding(
                section,
                getattr(draft, section),
                draft.grounding_for(section),  # type: ignore[arg-type]
                profile,
                insight,
                bank,
            )
        )
    conditional_ids = set(draft.section_3_grounding.evidence_dossier_ids).intersection(
        CONDITIONAL_GUIDANCE_DOSSIER_IDS
    )
    if len(conditional_ids) > 1:
        issues.append("section_3: at most one unobserved conditional recommendation is allowed")
    if conditional_ids:
        if not re.search(r"\bif\b", draft.section_3, re.I):
            issues.append(
                "section_3: unobserved conditional guidance requires an explicit if condition"
            )
        by_id = {item.dossier_id: item for item in bank.dossiers}
        relevant_metrics = {
            metric_id
            for dossier_id in conditional_ids
            for metric_id in by_id[dossier_id].supported_outcome_dimensions
        }
        evaluation_metrics = {
            anchor.metric_id for anchor in draft.section_4_grounding.case_anchors
        }
        if not relevant_metrics.intersection(evaluation_metrics):
            issues.append(
                "section_4: conditional recommendation requires a visible outcome metric "
                "linked to its Evidence"
            )
        current_evaluation_metrics = {
            anchor.metric_id
            for anchor in draft.section_4_grounding.case_anchors
            if anchor.statistic.value in _CURRENT_BASELINE_STATISTICS
        }
        if not relevant_metrics.intersection(current_evaluation_metrics):
            issues.append(
                "section_4: conditional recommendation requires a current pre-trial "
                "multi-night baseline for an Evidence-supported outcome"
            )
        if _HISTORICAL_SUCCESS_TARGET.search(
            draft.section_4
        ) and not _NEGATED_HISTORICAL_TARGET.search(draft.section_4):
            issues.append(
                "section_4: historical values may provide context but cannot define "
                "success for a conditional trial; compare sustained change with the "
                "current pre-trial multi-night baseline"
            )
        if conditional_ids.intersection(
            {"caffeine-timing-meta", "alcohol-sleep-context-meta"}
        ) and re.search(
            r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve)"
            r"[\s-]*(?:hours?|hrs?|minutes?|mins?)\s+"
            r"(?:earlier|before|prior to)",
            draft.section_3,
            re.I,
        ):
            issues.append(
                "section_3: caffeine/alcohol guidance cannot invent a universal "
                "numeric timing cutoff"
            )
        if "clinician-guided-melatonin" in conditional_ids:
            clinician = re.search(
                r"\b(?:clinician|doctor|physician|healthcare professional|sleep specialist)\b",
                draft.section_3,
                re.I,
            )
            discussion = re.search(
                r"\b(?:discuss|ask|consult|talk (?:to|with))\b",
                draft.section_3,
                re.I,
            )
            self_directed = re.search(
                r"\b(?:start|take|try|use|begin|consider)\s+(?:taking\s+|using\s+)?melatonin\b",
                draft.section_3,
                re.I,
            )
            if not clinician or not discussion or self_directed:
                issues.append(
                    "section_3: melatonin may only be framed as something to discuss "
                    "with a qualified clinician, never as self-directed use"
                )
    return issues

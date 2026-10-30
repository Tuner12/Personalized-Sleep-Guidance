"""Contracts for the redesigned raw-data -> Profile -> packet boundary.

There is intentionally no high/moderate/low reliability label.  Each limitation
must instead name a concrete reason: sample size, a device-derived field, circular
clock arithmetic, a source inconsistency, or the non-causal nature of an association.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetricStatus(StrEnum):
    COMPUTED = "computed"
    NO_DATA = "no_data"


class WindowTier(StrEnum):
    INSUFFICIENT = "insufficient"
    EXPLORATORY = "exploratory"
    ADEQUATE = "adequate"
    ROBUST = "robust"


class DataIssue(ContractModel):
    """One concrete source or arithmetic issue, never a composite confidence score."""

    code: str
    scope: Literal["case", "night"]
    explanation: str
    sleep_date: date | None = None
    observed_difference_min: float | None = None


class DailyObservation(ContractModel):
    sleep_date: date
    day_index: int = Field(ge=1)
    day_of_week: str
    is_weekend: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    source_issues: list[DataIssue] = Field(default_factory=list)


class MetricSummary(ContractModel):
    metric_id: str
    unit: str
    status: MetricStatus = MetricStatus.COMPUTED
    n: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    sd: float | None = Field(default=None, ge=0)
    mad: float | None = Field(default=None, ge=0)
    iqr: float | None = Field(default=None, ge=0)
    p10: float | None = None
    p90: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    first_value: float | None = None
    last_value: float | None = None
    caveats: list[str] = Field(default_factory=list)


class ComparisonWindow(ContractModel):
    start_date: date
    end_date: date
    n: int = Field(ge=1)
    calendar_span_days: int = Field(ge=1)
    center: float
    mad: float = Field(ge=0)
    iqr: float = Field(ge=0)


class EpisodicDeviation(ContractModel):
    """One unusual night retained for interpretation, not a clinical abnormality."""

    sleep_date: date
    day_index: int = Field(ge=1)
    segment: Literal["earlier", "middle", "recent"]
    value: float
    segment_center: float
    difference_from_segment_center: float


class LongitudinalSummary(ContractModel):
    metric_id: str
    unit: str
    n: int = Field(ge=2)
    span_days: int = Field(ge=1)
    sampling_density_pct: float = Field(ge=0, le=100)
    window_tier: WindowTier
    earlier_window: ComparisonWindow | None = None
    middle_window: ComparisonWindow | None = None
    recent_window: ComparisonWindow | None = None
    recent_minus_earlier: float | None = None
    recent_minus_earlier_mad: float | None = None
    slope_per_7_days: float | None = None
    comparison_direction_agrees_with_slope: bool | None = None
    episodic_deviations: list[EpisodicDeviation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class WeekpartComparison(ContractModel):
    metric_id: str
    unit: str
    weekday_n: int = Field(ge=0)
    weekend_n: int = Field(ge=0)
    weekday_center: float | None = None
    weekend_center: float | None = None
    weekend_minus_weekday: float | None = None
    reportable: bool = False
    caveats: list[str] = Field(default_factory=list)


class DailyContextAssociation(ContractModel):
    context_metric_id: str
    sleep_metric_id: str
    unit_context: str
    unit_sleep: str
    pairing: Literal["previous_calendar_day_to_sleep_date"]
    paired_nights: int = Field(ge=2)
    spearman_rho: float = Field(ge=-1, le=1)
    high_minus_low_sleep: float | None = None
    reportable: bool = False
    caveats: list[str] = Field(default_factory=list)


class SleepProfile(ContractModel):
    """Complete deterministic profile retained for audit."""

    case_id: str
    age_at_case_start_years: int | None = None
    sex: str | None = None
    valid_main_sleep_nights: int = Field(ge=0)
    calendar_span_days: int = Field(ge=0)
    coverage_pct: float = Field(ge=0, le=100)
    daily_observations: list[DailyObservation] = Field(default_factory=list)
    metric_summaries: list[MetricSummary] = Field(default_factory=list)
    longitudinal_summaries: list[LongitudinalSummary] = Field(default_factory=list)
    weekday_weekend_comparisons: list[WeekpartComparison] = Field(default_factory=list)
    daily_context_associations: list[DailyContextAssociation] = Field(default_factory=list)
    data_issues: list[DataIssue] = Field(default_factory=list)
    interpretation_caveats: list[str] = Field(default_factory=list)


class InsightWindow(ContractModel):
    start_date: date
    end_date: date
    n: int = Field(ge=1)
    center: float
    mad: float = Field(ge=0)
    iqr: float = Field(ge=0)


class TemporalTrajectoryMetric(ContractModel):
    metric_id: str
    unit: str
    window_size_nights: int = Field(ge=1)
    windows: list[InsightWindow] = Field(default_factory=list)


class RecentChangeMetric(ContractModel):
    metric_id: str
    unit: str
    recent_7_nights: InsightWindow | None = None
    previous_7_nights: InsightWindow | None = None
    earlier_history: InsightWindow | None = None
    recent_minus_previous: float | None = None
    recent_minus_earlier_history: float | None = None


class LatestNightMetric(ContractModel):
    metric_id: str
    unit: str
    sleep_date: date
    day_index: int = Field(ge=1)
    latest_value: float
    prior_7_nights: InsightWindow | None = None
    earlier_history: InsightWindow | None = None
    latest_minus_prior_7: float | None = None
    latest_minus_earlier_history: float | None = None


class RecordEvent(ContractModel):
    metric_id: str
    unit: str
    sleep_date: date
    day_index: int = Field(ge=1)
    value: float
    full_record_center: float
    difference_from_full_record_center: float


class LongitudinalInsight(ContractModel):
    """Deterministic five-view longitudinal facts for every available metric."""

    case_id: str
    whole_record_summary: list[MetricSummary] = Field(default_factory=list)
    rolling_7_night_trajectory: list[TemporalTrajectoryMetric] = Field(default_factory=list)
    recent_7_night_comparison: list[RecentChangeMetric] = Field(default_factory=list)
    latest_night_context: list[LatestNightMetric] = Field(default_factory=list)
    fluctuation_points: list[RecordEvent] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_view_names(cls, value):
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        aliases = {
            "typical_state": "whole_record_summary",
            "full_record_summary": "whole_record_summary",
            "temporal_trajectory": "rolling_7_night_trajectory",
            "recent_change": "recent_7_night_comparison",
            "latest_night": "latest_night_context",
            "record_events": "fluctuation_points",
            "dated_record_events": "fluctuation_points",
        }
        for old, new in aliases.items():
            if old in migrated:
                migrated.setdefault(new, migrated.pop(old))
        return migrated

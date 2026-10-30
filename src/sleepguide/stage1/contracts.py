"""Minimal contracts for Stage 1.

The contracts deliberately separate bibliographic audit data from the compact card text
used to generate a recommendation.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, model_validator

from sleepguide.core.contracts import ContractModel
from sleepguide.core.recommendation import CaseMetricAnchor, ReaderRecommendation

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

STAGE1_SCHEMA_VERSION = "stage1-1.0"
OBSERVATION_CONTRACT_ID = "case-observation-1.0"


class EvidenceKind(StrEnum):
    REFERENCE = "reference"
    EXPLANATION = "explanation"
    GUIDANCE = "guidance"
    BOUNDARY = "boundary"


class EvidenceReferenceValue(ContractModel):
    metric_id: str
    comparator: str
    value: float
    unit: str
    label: str


class CaseObservation(ContractModel):
    case_id: str
    primary_observation: str = Field(
        min_length=1,
        description="The main repeated Profile pattern relevant to Evidence selection.",
    )
    supporting_observations: list[str] = Field(default_factory=list, max_length=3)
    context_observation: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "An observed counterpoint or boundary that materially changes interpretation; "
            "omit it when no such context is needed."
        ),
    )
    relevant_metric_ids: list[str] = Field(min_length=1)
    anchors: list[CaseMetricAnchor] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_metrics(self) -> "CaseObservation":
        metrics = self.relevant_metric_ids
        if len(metrics) != len(set(metrics)):
            raise ValueError("Observation relevant_metric_ids must be unique")
        return self


class EvidenceCard(ContractModel):
    evidence_key: str
    title: str
    kind: EvidenceKind
    meaning: str = Field(min_length=1)
    next_step: str | None = None
    canonical_action_title: str | None = None
    action_title_variants: list[str] = Field(default_factory=list, max_length=3)
    follow_up: str | None = None
    use_when: str = Field(min_length=1)
    evaluate_with: list[str] = Field(default_factory=list)
    reference_values: list[EvidenceReferenceValue] = Field(default_factory=list)
    condition: str | None = None
    constraints: list[str] = Field(default_factory=list)
    source_dossier_ids: list[str] = Field(min_length=1)
    runtime_enabled: bool = True

    @model_validator(mode="after")
    def validate_card(self) -> "EvidenceCard":
        if not _SLUG.fullmatch(self.evidence_key):
            raise ValueError("evidence_key must be a readable lowercase slug")
        if self.kind is EvidenceKind.GUIDANCE and not self.next_step:
            raise ValueError("guidance Evidence requires next_step")
        if self.kind is EvidenceKind.REFERENCE and not self.reference_values:
            raise ValueError("reference Evidence requires a structured reference value")
        if self.condition and not self.next_step:
            raise ValueError("a conditional card requires next_step")
        if self.next_step and not self.canonical_action_title:
            raise ValueError("action-permitting Evidence requires a canonical action title")
        if self.next_step and not self.action_title_variants:
            raise ValueError("action-permitting Evidence requires at least one title variant")
        if not self.next_step and (
            self.canonical_action_title or self.action_title_variants
        ):
            raise ValueError("non-action Evidence cannot define action titles")
        allowed_titles = [
            title
            for title in [self.canonical_action_title, *self.action_title_variants]
            if title is not None
        ]
        if len({title.casefold() for title in allowed_titles}) != len(allowed_titles):
            raise ValueError("canonical and variant action titles must be unique")
        if len(self.source_dossier_ids) != len(set(self.source_dossier_ids)):
            raise ValueError("source_dossier_ids must be unique")
        return self

    @property
    def allowed_action_titles(self) -> list[str]:
        return [
            title
            for title in [self.canonical_action_title, *self.action_title_variants]
            if title is not None
        ]


class EvidenceCardLibrary(ContractModel):
    release_id: str
    source_bank_release_id: str
    cards: list[EvidenceCard]

    @model_validator(mode="after")
    def unique_cards(self) -> "EvidenceCardLibrary":
        keys = [card.evidence_key for card in self.cards]
        if len(keys) != len(set(keys)):
            raise ValueError("Evidence card library contains duplicate keys")
        action_titles = [
            title.casefold()
            for card in self.cards
            for title in card.allowed_action_titles
        ]
        if len(action_titles) != len(set(action_titles)):
            raise ValueError("action titles must be unique across Evidence cards")
        return self


class EvidenceSelectionItem(ContractModel):
    evidence_key: str
    case_relevance: str = Field(
        min_length=1,
        description=(
            "One natural sentence linking the repeated Case Observation pattern to this "
            "Evidence meaning; it is not the final recommendation."
        ),
    )
    matched_outcomes: list[str] = Field(default_factory=list)


class EvidenceSelection(ContractModel):
    case_id: str
    selected: list[EvidenceSelectionItem] = Field(default_factory=list, max_length=5)
    selection_summary: str = Field(
        min_length=1,
        description="Short natural reasoning chain explaining why the selected set is sufficient.",
    )

    @model_validator(mode="after")
    def unique_selection(self) -> "EvidenceSelection":
        keys = [item.evidence_key for item in self.selected]
        if len(keys) != len(set(keys)):
            raise ValueError("Evidence selection contains duplicate keys")
        return self


class Stage1Plan(ContractModel):
    observation: CaseObservation
    evidence_selection: EvidenceSelection

    @model_validator(mode="after")
    def same_case(self) -> "Stage1Plan":
        if self.observation.case_id != self.evidence_selection.case_id:
            raise ValueError("Observation and Evidence Selection must name the same case")
        return self


class EvidenceReferenceAnchor(ContractModel):
    evidence_key: str
    metric_id: str
    comparator: str
    value: float
    unit: str
    label: str


class SectionGrounding(ContractModel):
    evidence_keys: list[str] = Field(default_factory=list)
    case_anchors: list[CaseMetricAnchor] = Field(default_factory=list)
    outcome_metrics: list[str] = Field(default_factory=list)
    evidence_reference_values: list[EvidenceReferenceAnchor] = Field(default_factory=list)


class RecommendationDraft(ContractModel):
    case_id: str
    section_1: str = Field(min_length=1)
    section_2: str = Field(min_length=1)
    section_3: str = Field(min_length=1)
    section_4: str = Field(min_length=1)
    section_1_grounding: SectionGrounding
    section_2_grounding: SectionGrounding
    section_3_grounding: SectionGrounding
    section_4_grounding: SectionGrounding

    def render_sections(self) -> dict[str, str]:
        return {
            "section_1": self.section_1.strip(),
            "section_2": self.section_2.strip(),
            "section_3": self.section_3.strip(),
            "section_4": self.section_4.strip(),
        }


class ReverseIssueType(StrEnum):
    FACTUAL = "factual"
    EVIDENCE = "evidence"
    LANGUAGE = "language"
    REASONING = "reasoning"


class ReverseIssue(ContractModel):
    issue_type: ReverseIssueType
    section: str
    reason: str = Field(min_length=1)
    repair_instruction: str = Field(min_length=1)


class ReverseSectionReview(ContractModel):
    section_text: str
    recovered_evidence_keys: list[str] = Field(default_factory=list)
    issues: list[ReverseIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_recovered_keys(self) -> "ReverseSectionReview":
        if len(self.recovered_evidence_keys) != len(set(self.recovered_evidence_keys)):
            raise ValueError("Reverse recovered Evidence keys must be unique")
        return self


class BlindReverseReview(ContractModel):
    case_id: str
    reproduced_recommendation: ReaderRecommendation
    section_1: ReverseSectionReview
    section_2: ReverseSectionReview
    section_3: ReverseSectionReview
    section_4: ReverseSectionReview
    verdict: str

    def sections(self) -> dict[str, ReverseSectionReview]:
        return {
            "section_1": self.section_1,
            "section_2": self.section_2,
            "section_3": self.section_3,
            "section_4": self.section_4,
        }

    @model_validator(mode="after")
    def valid_verdict(self) -> "BlindReverseReview":
        expected = "pass" if not any(x.issues for x in self.sections().values()) else "revise"
        if self.verdict != expected:
            raise ValueError(f"verdict must be {expected!r}")
        return self


class ReverseComparison(ContractModel):
    section: str
    writer_used_evidence_keys: list[str] = Field(default_factory=list)
    reverse_recovered_evidence_keys: list[str] = Field(default_factory=list)
    exact_match: bool
    writer_only: list[str] = Field(default_factory=list)
    reverse_only: list[str] = Field(default_factory=list)


class ReverseResult(ContractModel):
    review: BlindReverseReview
    comparisons: list[ReverseComparison]
    issues: list[ReverseIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.review.verdict == "pass" and not self.issues

"""Loader for the evidence dossiers whose reference values the scorer excludes from Value F1."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from sleepguide.core.contracts import ContractModel

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

CONDITIONAL_GUIDANCE_DOSSIER_IDS = frozenset(
    {
        "alcohol-sleep-context-meta",
        "caffeine-timing-meta",
        "circadian-phase-treatment-boundary",
        "clinician-guided-melatonin",
        "conditional-light-and-screen-adjustment",
        "conditional-sleep-environment-adjustment",
        "quiet-wind-down-experiment",
        "stimulus-control-professional-option",
    }
)


class EvidenceSource(ContractModel):
    source_id: str
    title: str
    organization: str
    year: int = Field(ge=1990, le=2100)
    url: str
    locator: str
    source_tier: str


class EvidenceReferenceValue(ContractModel):
    metric_id: str
    comparator: str
    value: float
    unit: str
    label: str
    use_as: str


class EvidenceLevel(StrEnum):
    """Primary knowledge role of an Evidence dossier."""

    REFERENCE = "reference"
    EXPLANATION = "explanation"
    GUIDANCE = "guidance"
    LIMITATION = "limitation"


class EvidenceDossier(ContractModel):
    dossier_id: str
    title: str
    claim_summary: str
    evidence_basis: str
    population: str
    applicability: str
    limitations: list[str] = Field(min_length=1)
    case_dimensions: list[str] = Field(default_factory=list)
    supported_outcome_dimensions: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel
    reference_values: list[EvidenceReferenceValue] = Field(default_factory=list)
    prohibited_uses: list[str] = Field(default_factory=list)
    source: EvidenceSource
    runtime_enabled: bool = True

    @model_validator(mode="after")
    def validate_readable_identity(self) -> EvidenceDossier:
        if not _SLUG.fullmatch(self.dossier_id) or self.dossier_id.startswith("e-"):
            raise ValueError("dossier_id must be a readable lowercase slug without E_ prefix")
        if self.title.isupper() or self.title.startswith("E_"):
            raise ValueError("dossier title must be reader-facing rather than an engineering ID")
        if self.evidence_level is EvidenceLevel.REFERENCE and not self.reference_values:
            raise ValueError("reference evidence must contain a structured reference value")
        unknown_reference_metrics = {
            item.metric_id for item in self.reference_values
        } - set(self.case_dimensions)
        if unknown_reference_metrics:
            raise ValueError(
                "reference metrics must also be declared as case dimensions: "
                f"{sorted(unknown_reference_metrics)}"
            )
        return self


class ReadableEvidenceBank(ContractModel):
    release_id: str
    criteria_id: str
    dossiers: list[EvidenceDossier]
    all_sources_verified: bool = True
    all_dossiers_verified: bool = True

    @model_validator(mode="after")
    def validate_bank(self) -> ReadableEvidenceBank:
        ids = [item.dossier_id for item in self.dossiers]
        titles = [item.title.casefold() for item in self.dossiers]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence bank contains duplicate dossier IDs")
        if len(titles) != len(set(titles)):
            raise ValueError("evidence bank contains duplicate reader-facing titles")
        if not self.all_sources_verified or not self.all_dossiers_verified:
            raise ValueError("runtime requires a fully verified evidence release")
        conditional_by_id = {
            item.dossier_id: item
            for item in self.dossiers
            if item.runtime_enabled
            and item.dossier_id in CONDITIONAL_GUIDANCE_DOSSIER_IDS
        }
        missing_outcomes = sorted(
            dossier_id
            for dossier_id, item in conditional_by_id.items()
            if not item.supported_outcome_dimensions
        )
        if missing_outcomes:
            raise ValueError(
                "runtime conditional Evidence requires supported outcome dimensions: "
                f"{missing_outcomes}"
            )
        invalid_outcomes = {
            dossier_id: sorted(
                set(item.supported_outcome_dimensions) - set(item.case_dimensions)
            )
            for dossier_id, item in conditional_by_id.items()
            if set(item.supported_outcome_dimensions) - set(item.case_dimensions)
        }
        if invalid_outcomes:
            raise ValueError(
                "conditional outcome dimensions must be available case dimensions: "
                f"{invalid_outcomes}"
            )
        return self


def load_readable_evidence_bank(path: str | Path | None = None) -> ReadableEvidenceBank:
    default = (
        Path(__file__).resolve().parents[3] / "evidence_bank" / "scoring" / "reference_dossiers.json"
    )
    return ReadableEvidenceBank.model_validate_json(
        (Path(path) if path is not None else default).read_text(encoding="utf-8")
    )


def model_facing_dossier(dossier: EvidenceDossier) -> dict:
    """Expose compact evidence content without assigning it to a Section."""

    return {
        "dossier_id": dossier.dossier_id,
        "title": dossier.title,
        "evidence": dossier.claim_summary,
        "population": dossier.population,
        "applicability": dossier.applicability,
        "limitations": dossier.limitations,
        "evidence_level": dossier.evidence_level.value,
        "case_dimensions": dossier.case_dimensions,
        "supported_outcome_dimensions": dossier.supported_outcome_dimensions,
        "reference_values": [item.model_dump(mode="json") for item in dossier.reference_values],
        "prohibited_uses": dossier.prohibited_uses,
        "source": {
            "title": dossier.source.title,
            "organization": dossier.source.organization,
            "year": dossier.source.year,
        },
    }

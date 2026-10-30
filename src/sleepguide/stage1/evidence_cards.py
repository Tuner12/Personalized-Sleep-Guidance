"""Load the Evidence Bank's runtime cards and render them for the Evidence and Composer Agents."""

from __future__ import annotations

import json
from pathlib import Path

from sleepguide.core.evidence import (
    EvidenceDossier,
    load_readable_evidence_bank,
)
from sleepguide.stage1.contracts import EvidenceCard, EvidenceCardLibrary, EvidenceSelection


def load_generation_card_library(
    path: str | Path | None = None,
    *,
    source_bank_path: str | Path | None = None,
    source_extensions_path: str | Path | None = None,
) -> EvidenceCardLibrary:
    root = Path(__file__).resolve().parents[3]
    card_path = (
        Path(path)
        if path is not None
        else root / "evidence_bank" / "cards.json"
    )
    library = EvidenceCardLibrary.model_validate_json(card_path.read_text(encoding="utf-8"))
    source_bank = load_readable_evidence_bank(source_bank_path)
    if library.source_bank_release_id != source_bank.release_id:
        raise ValueError(
            "generation-card source release differs from the verified source bank: "
            f"{library.source_bank_release_id!r} != {source_bank.release_id!r}"
        )
    extension_path = (
        Path(source_extensions_path)
        if source_extensions_path is not None
        else root / "evidence_bank" / "source_extensions.json"
    )
    extension_dossiers: list[EvidenceDossier] = []
    if extension_path.exists():
        extension_payload = json.loads(extension_path.read_text(encoding="utf-8"))
        if extension_payload.get("extends_source_bank_release_id") != source_bank.release_id:
            raise ValueError("source extension release does not match the verified source bank")
        extension_dossiers = [
            EvidenceDossier.model_validate(item)
            for item in extension_payload.get("dossiers", [])
        ]
    source_ids = {
        item.dossier_id for item in [*source_bank.dossiers, *extension_dossiers]
    }
    if len(source_ids) != len(source_bank.dossiers) + len(extension_dossiers):
        raise ValueError("source extension duplicates a source dossier ID")
    missing = {
        card.evidence_key: sorted(set(card.source_dossier_ids) - source_ids)
        for card in library.cards
        if set(card.source_dossier_ids) - source_ids
    }
    if missing:
        raise ValueError(f"generation cards reference missing source dossiers: {missing}")
    return library


def writer_card(card: EvidenceCard) -> dict:
    """Return only the information that can improve the four reader-facing Sections."""

    return {
        "evidence_key": card.evidence_key,
        "evidence_role": card.kind.value,
        "analysis_support": card.meaning,
        "recommendation_support": card.next_step,
        "canonical_action_title": card.canonical_action_title,
        "allowed_action_titles": card.allowed_action_titles,
        "follow_up_support": card.follow_up,
        "select_when": card.use_when,
        "outcome_metrics": card.evaluate_with,
        "reference_values": [
            item.model_dump(mode="json") for item in card.reference_values
        ],
        "required_condition": card.condition,
        "do_not_claim": card.constraints,
    }


def build_case_evidence(
    selection: EvidenceSelection,
    library: EvidenceCardLibrary,
) -> dict:
    """Combine the natural case link and verified card semantics into Evidence E."""

    by_key = {card.evidence_key: card for card in library.cards}
    return {
        "case_id": selection.case_id,
        "selection_summary": selection.selection_summary,
        "items": [
            {
                **writer_card(by_key[item.evidence_key]),
                "case_relevance": item.case_relevance,
                "matched_outcomes": item.matched_outcomes,
            }
            for item in selection.selected
        ],
    }

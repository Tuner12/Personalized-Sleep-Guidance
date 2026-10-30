"""Run Stage 1 for one case: up to five Evidence -> Composer -> Reflection cycles.

A case is `certified` when the Reflection Agent's independently recovered evidence matches the
Evidence Agent's selection; otherwise its feedback drives the next cycle, up to five cycles.
Intermediate artifacts are saved so a case can resume.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from sleepguide.llm import LLMClient
from sleepguide.core.contracts import LongitudinalInsight, SleepProfile
from sleepguide.core.packet import generation_model_input
from sleepguide.stage1 import (
    build_case_evidence,
    load_generation_card_library,
    writer_card,
)
from sleepguide.stage1.agents import (
    plan_case,
    polish_recommendation,
    review_blind_reverse,
    write_recommendation,
)
from sleepguide.stage1.contracts import (
    OBSERVATION_CONTRACT_ID,
    STAGE1_SCHEMA_VERSION,
    EvidenceSelection,
    RecommendationDraft,
    ReverseIssueType,
    Stage1Plan,
)


def load_profile_packet(profiles_dir: Path, case_id: str) -> tuple[SleepProfile, LongitudinalInsight]:
    """Read the Profile Agent's packet P for one case, validated against the contracts."""
    packet = json.loads((profiles_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    return (SleepProfile.model_validate(packet["profile"]),
            LongitudinalInsight.model_validate(packet["longitudinal_insight"]))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", type=Path, required=True,
                        help="Profile Agent output: one <case_id>.json per case")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--providers", type=Path, default=Path("configs/providers.yaml"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--max-cycles", type=int, default=5,
                        help="Evidence -> Composer -> Reflection cycles per case (paper: 5)")
    parser.add_argument("--max-stage-retries", type=int, default=3)
    parser.add_argument("--skip-reverse", action="store_true")
    parser.add_argument(
        "--generation-cards",
        type=Path,
        default=Path("evidence_bank/cards.json"),
    )
    parser.add_argument(
        "--source-bank",
        type=Path,
        default=Path("evidence_bank/sources.json"),
    )
    return parser.parse_args()


def _client(path: Path, name: str | None) -> LLMClient:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    provider_name = name or config["default_provider"]
    spec = config["providers"][provider_name]
    return LLMClient(
        provider=provider_name,
        command=spec["command"],
        args=tuple(spec.get("args", [])),
        protocol=spec.get("protocol", "stdin_json"),
        timeout_s=spec.get("timeout_s", 600),
        protocol_attempts=spec.get("protocol_attempts", 2),
        transient_attempts=spec.get("transient_attempts", 4),
        transient_backoff_s=spec.get("transient_backoff_s", 10.0),
        max_concurrent_calls=spec.get("max_concurrent_calls", 10),
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _feedback(issues: list[Any]) -> list[dict[str, Any]]:
    return [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else {"reason": str(item)}
        for item in issues
    ]


def _editable_sections(feedback: list[dict[str, Any]]) -> set[str]:
    sections: set[str] = set()
    for item in feedback:
        explicit = item.get("section")
        if explicit in {"section_1", "section_2", "section_3", "section_4"}:
            sections.add(explicit)
        reason = str(item.get("reason") or "")
        for match in re.finditer(r"\bsection[_ ]([1-4])\b", reason, re.I):
            sections.add(f"section_{match.group(1)}")
    return sections


def _successful_polish_for_writer(
    attempts: list[dict[str, Any]],
    revision: int,
    source_draft: RecommendationDraft,
) -> RecommendationDraft | None:
    """Restore polish only when it belongs to the exact validated Composer draft."""

    source_payload = source_draft.model_dump(mode="json")
    for item in reversed(attempts):
        if (
            item.get("stage") == "language_polish"
            and item.get("revision") == revision
            and not item.get("issues")
            and item.get("source_recommendation_draft") == source_payload
            and item.get("polished_recommendation_draft")
        ):
            return RecommendationDraft.model_validate(
                item["polished_recommendation_draft"]
            )
    return None


def run(
    args: argparse.Namespace,
    *,
    planner_client: Any | None = None,
    writer_client: Any | None = None,
    polish_client: Any | None = None,
    reverse_client: Any | None = None,
) -> dict[str, Any]:
    profile, insight = load_profile_packet(args.profiles_dir, args.case_id)
    library = load_generation_card_library(
        args.generation_cards, source_bank_path=args.source_bank
    )
    generation_evidence_bank = [
        writer_card(card) for card in library.cards if card.runtime_enabled
    ]
    model_input = generation_model_input(
        profile.model_dump(mode="json"), insight.model_dump(mode="json")
    )
    planner_client = planner_client or _client(args.providers, args.provider)
    writer_client = writer_client or _client(args.providers, args.provider)
    polish_client = polish_client or writer_client
    if not getattr(args, "skip_reverse", False):
        reverse_client = reverse_client or _client(args.providers, args.provider)
    case_dir = args.output_dir / args.case_id
    state_path = case_dir / "stage1_state.json"

    attempts: list[dict[str, Any]] = []
    plan: Stage1Plan | None = None
    draft: RecommendationDraft | None = None
    source_draft_for_polish: RecommendationDraft | None = None
    polish_complete = False
    plan_feedback: list[dict[str, Any]] | None = None
    previous_plan: Stage1Plan | None = None
    writer_feedback: list[dict[str, Any]] | None = None
    editable_sections: set[str] | None = None
    if state_path.exists():
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
            attempts = list(prior_state.get("attempts", []))
        except json.JSONDecodeError:
            attempts = []
        if attempts and prior_state.get("observation_contract") != OBSERVATION_CONTRACT_ID:
            raise RuntimeError(
                "Existing stage1_state.json uses an incompatible Observation contract. "
                "Use a new output directory for compact Case Observation v3."
            )
        for item in reversed(attempts):
            if (
                item.get("stage") == "planning"
                and not item.get("issues")
                and item.get("plan")
            ):
                plan = Stage1Plan.model_validate(item["plan"])
                break
        if plan is not None:
            plan_revision = max(
                item.get("revision", -1)
                for item in attempts
                if item.get("stage") == "planning"
                and not item.get("issues")
                and item.get("plan")
            )
            for item in reversed(attempts):
                if (
                    item.get("stage") == "writer"
                    and item.get("revision") == plan_revision
                    and not item.get("issues")
                    and item.get("recommendation_draft")
                ):
                    draft = RecommendationDraft.model_validate(
                        item["recommendation_draft"]
                    )
                    source_draft_for_polish = draft
                    break
            if source_draft_for_polish is not None:
                restored_polish = _successful_polish_for_writer(
                    attempts, plan_revision, source_draft_for_polish
                )
                if restored_polish is not None:
                    draft = restored_polish
                    polish_complete = True
        else:
            last_failed_plan = next(
                (
                    item
                    for item in reversed(attempts)
                    if item.get("stage") == "planning"
                    and item.get("plan")
                    and item.get("issues")
                ),
                None,
            )
            if last_failed_plan is not None:
                previous_plan = Stage1Plan.model_validate(last_failed_plan["plan"])
                plan_feedback = _feedback(list(last_failed_plan["issues"]))
        if plan is not None and draft is None:
            last_failed_draft = next(
                (
                    item
                    for item in reversed(attempts)
                    if item.get("stage") == "writer"
                    and item.get("revision") == plan_revision
                    and item.get("recommendation_draft")
                    and item.get("issues")
                ),
                None,
            )
            if last_failed_draft is not None:
                draft = RecommendationDraft.model_validate(
                    last_failed_draft["recommendation_draft"]
                )
                writer_feedback = _feedback(list(last_failed_draft["issues"]))
                editable_sections = _editable_sections(writer_feedback) or None

    starting_revision = max(
        [int(item.get("revision", 0)) for item in attempts] or [0]
    )
    # Cycles are numbered 0..max_cycles-1 across resumes, so a case never exceeds max_cycles.
    for revision in range(starting_revision, args.max_cycles):
        if plan is None:
            for stage_retry in range(args.max_stage_retries):
                plan, planning_issues = plan_case(
                    planner_client,
                    profile,
                    insight,
                    library,
                    previous_plan=previous_plan,
                    review_feedback=plan_feedback,
                )
                attempts.append(
                    {
                        "revision": revision,
                        "stage": "planning",
                        "stage_retry": stage_retry,
                        "review_feedback": plan_feedback,
                        "plan": plan.model_dump(mode="json"),
                        "issues": planning_issues,
                    }
                )
                _write(
                    state_path,
                    {
                        "status": "running",
                        "observation_contract": OBSERVATION_CONTRACT_ID,
                        "case_id": args.case_id,
                        "current_stage": "planning",
                        "attempts": attempts,
                    },
                )
                if not planning_issues:
                    break
                previous_plan = plan
                plan_feedback = _feedback(planning_issues)
                plan = None
            if plan is None:
                break

        if draft is not None and not writer_feedback:
            writer_issues = []
        else:
            writer_issues = ["writer has not run"]
        for stage_retry in range(args.max_stage_retries):
            if draft is not None and not writer_feedback:
                break
            previous = draft if writer_feedback and editable_sections else None
            draft, writer_issues = write_recommendation(
                writer_client,
                profile,
                insight,
                plan.observation,
                plan.evidence_selection,
                library,
                previous_draft=previous,
                review_feedback=writer_feedback,
                editable_sections=editable_sections,
            )
            attempts.append(
                {
                    "revision": revision,
                    "stage": "writer",
                    "stage_retry": stage_retry,
                    "review_feedback": writer_feedback,
                    "recommendation_draft": draft.model_dump(mode="json"),
                    "issues": writer_issues,
                }
            )
            _write(
                state_path,
                {
                    "status": "running",
                    "observation_contract": OBSERVATION_CONTRACT_ID,
                    "case_id": args.case_id,
                    "current_stage": "writer",
                    "attempts": attempts,
                },
            )
            if not writer_issues:
                source_draft_for_polish = draft
                polish_complete = False
                break
            writer_feedback = _feedback(writer_issues)
            editable_sections = _editable_sections(writer_feedback) or None
        if draft is None or writer_issues:
            break

        polish_issues: list[str] = [] if polish_complete else ["language polish has not run"]
        if not polish_complete:
            source_draft = source_draft_for_polish or draft
            previous_polished_draft: RecommendationDraft | None = None
            polish_feedback: list[dict[str, Any]] | None = None
            for polish_retry in range(args.max_stage_retries):
                try:
                    polished_draft, polish_issues = polish_recommendation(
                        polish_client,
                        profile,
                        insight,
                        plan.evidence_selection,
                        library,
                        source_draft,
                        previous_polished_draft=previous_polished_draft,
                        review_feedback=polish_feedback,
                    )
                except Exception as error:
                    # The Composer draft has already passed the full deterministic contract.
                    # If the separate language call cannot produce a valid structured
                    # response after its own retries, preserve that validated draft and
                    # record the protocol fallback instead of losing a valid case.
                    draft = source_draft
                    polish_issues = []
                    attempts.append(
                        {
                            "revision": revision,
                            "stage": "language_polish_fallback",
                            "stage_retry": polish_retry,
                            "review_feedback": polish_feedback,
                            "source_recommendation_draft": source_draft.model_dump(mode="json"),
                            "polished_recommendation_draft": source_draft.model_dump(mode="json"),
                            "issues": [],
                            "fallback_reason": str(error),
                        }
                    )
                    _write(
                        state_path,
                        {
                            "status": "running",
                            "observation_contract": OBSERVATION_CONTRACT_ID,
                            "case_id": args.case_id,
                            "current_stage": "language_polish_fallback",
                            "attempts": attempts,
                        },
                    )
                    polish_complete = True
                    break
                attempts.append(
                    {
                        "revision": revision,
                        "stage": "language_polish",
                        "stage_retry": polish_retry,
                        "review_feedback": polish_feedback,
                        "source_recommendation_draft": source_draft.model_dump(mode="json"),
                        "polished_recommendation_draft": polished_draft.model_dump(mode="json"),
                        "issues": polish_issues,
                    }
                )
                _write(
                    state_path,
                    {
                        "status": "running",
                        "observation_contract": OBSERVATION_CONTRACT_ID,
                        "case_id": args.case_id,
                        "current_stage": "language_polish",
                        "attempts": attempts,
                    },
                )
                if not polish_issues:
                    draft = polished_draft
                    polish_complete = True
                    break
                previous_polished_draft = polished_draft
                polish_feedback = _feedback(polish_issues)
        if polish_issues:
            break

        if getattr(args, "skip_reverse", False):
            selection: EvidenceSelection = plan.evidence_selection
            result = {
                "schema_version": STAGE1_SCHEMA_VERSION,
                "observation_contract": OBSERVATION_CONTRACT_ID,
                "input_contract": "profile_agent_packet",
                "profile_packet": str(args.profiles_dir / f"{args.case_id}.json"),
                "status": "validated_no_reverse",
                "case_id": args.case_id,
                "evidence_release_id": library.release_id,
                "profile": profile.model_dump(mode="json"),
                "longitudinal_insight": insight.model_dump(mode="json"),
                "model_input": model_input,
                "observation": plan.observation.model_dump(mode="json"),
                "evidence_selection": selection.model_dump(mode="json"),
                "generation_evidence_bank": generation_evidence_bank,
                "evidence": build_case_evidence(selection, library),
                "recommendation_draft": draft.model_dump(mode="json"),
                "recommendation": draft.render_sections(),
                "reverse_review": {
                    "run": False,
                    "reason": "reverse review skipped (--skip-reverse)",
                },
                "draft_revision": revision,
                "attempts": attempts,
            }
            _write(state_path, {**result, "current_stage": "complete"})
            return result

        if reverse_client is None:
            raise RuntimeError("the Reflection Agent needs a client unless --skip-reverse is set")
        reverse = review_blind_reverse(
            reverse_client, profile, insight, draft, library
        )
        attempts.append(
            {
                "revision": revision,
                "stage": "blind_reverse",
                "reverse_result": reverse.model_dump(mode="json"),
            }
        )
        _write(
            state_path,
            {
                "status": "running",
                "observation_contract": OBSERVATION_CONTRACT_ID,
                "case_id": args.case_id,
                "current_stage": "blind_reverse",
                "attempts": attempts,
            },
        )
        if reverse.passed:
            selection: EvidenceSelection = plan.evidence_selection
            result = {
                "schema_version": STAGE1_SCHEMA_VERSION,
                "observation_contract": OBSERVATION_CONTRACT_ID,
                "input_contract": "profile_agent_packet",
                "profile_packet": str(args.profiles_dir / f"{args.case_id}.json"),
                "status": "certified",
                "case_id": args.case_id,
                "evidence_release_id": library.release_id,
                "profile": profile.model_dump(mode="json"),
                "longitudinal_insight": insight.model_dump(mode="json"),
                "model_input": model_input,
                "observation": plan.observation.model_dump(mode="json"),
                "evidence_selection": selection.model_dump(mode="json"),
                "generation_evidence_bank": generation_evidence_bank,
                "evidence": build_case_evidence(selection, library),
                "recommendation_draft": draft.model_dump(mode="json"),
                "recommendation": draft.render_sections(),
                "reverse_review": reverse.model_dump(mode="json"),
                "draft_revision": revision,
                "attempts": attempts,
            }
            _write(state_path, {**result, "current_stage": "complete"})
            return result

        feedback = _feedback(reverse.issues)
        issue_types = {item.issue_type for item in reverse.issues}
        if issue_types.intersection({ReverseIssueType.EVIDENCE, ReverseIssueType.REASONING}):
            plan_feedback = feedback
            writer_feedback = feedback
            previous_plan = plan
            plan = None
            draft = None
            source_draft_for_polish = None
            polish_complete = False
            editable_sections = None
        else:
            writer_feedback = feedback
            editable_sections = _editable_sections(feedback) or None
            source_draft_for_polish = None
            polish_complete = False

    return {
        "schema_version": STAGE1_SCHEMA_VERSION,
        "observation_contract": OBSERVATION_CONTRACT_ID,
        "input_contract": "profile_agent_packet",
        "profile_packet": str(args.profiles_dir / f"{args.case_id}.json"),
        "status": "failed",
        "case_id": args.case_id,
        "evidence_release_id": library.release_id,
        "generation_evidence_bank": generation_evidence_bank,
        "profile": profile.model_dump(mode="json"),
        "longitudinal_insight": insight.model_dump(mode="json"),
        "model_input": model_input,
        "terminal_issue": "Stage 1 cycle or stage-retry budget exhausted",
        "attempts": attempts,
    }


def main() -> None:
    args = _args()
    result = run(args)
    case_dir = args.output_dir / args.case_id
    artifact = case_dir / "stage1_run.json"
    _write(artifact, result)
    if result["status"] in {"certified", "validated_no_reverse"}:
        _write(case_dir / "recommendation.json", result["recommendation"])
    print(f"status={result['status']}")
    print(f"artifact={artifact}")


if __name__ == "__main__":
    main()

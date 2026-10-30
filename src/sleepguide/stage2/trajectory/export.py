"""Step 1 of 6: export certified Stage-1 cases as Stage-2 task records.

Reads every case directory of a Stage-1 release (`stage1_run.json`) and writes one JSONL file
per task and split, all on the same case split:

    pure_sft            P -> R
    observation         P -> O
    evidence_selector   P, O, Evidence Bank -> E
    evidence_writer     P, O, E -> R

The later steps fold these into the single `[P; O; E; C; R]` trajectory sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sleepguide.llm import render_agent_request
from sleepguide.stage1 import STAGE1_SCHEMA_VERSION
from sleepguide.stage2.trajectory.task_prompts import (
    STAGE2_PROMPT_VERSION,
    stage2_prompt_catalog,
    stage2_task_prompt,
)

# Hash seed of the fixed train/test split; changing it changes which cases are held out.
SPLIT_SEED = "sleeprec-stage2-2026-08-25"


def _split(case_id: str, seed: str) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{case_id}".encode()).digest()[:8], "big"
    ) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reverse_review_ran(reverse: dict | None) -> bool:
    """Did the Reflection Agent's review run for this case?

    A skipped review is recorded as `{"run": false, "reason": ...}`; a completed one records
    its result (`review`, `comparisons`, `issues`) without a `run` key.
    """

    if not isinstance(reverse, dict):
        return False
    if "run" in reverse:
        return bool(reverse["run"])
    return "review" in reverse


def _observation_target(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_observation": value["primary_observation"],
        "supporting_observations": value["supporting_observations"],
        "context_observation": value.get("context_observation"),
        "relevant_metric_ids": value["relevant_metric_ids"],
    }


def _task_row(
    *,
    shared: dict[str, Any],
    task: str,
    input_payload: dict[str, Any],
    target: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    instruction = stage2_task_prompt(task)
    return {
        **shared,
        "task": task,
        "prompt_version": STAGE2_PROMPT_VERSION,
        "instruction": instruction,
        "input": input_payload,
        "rendered_input": render_agent_request(instruction, input_payload),
        "target": target,
        "audit": audit,
    }


def build(stage1_root: Path, out: Path, seed: str = SPLIT_SEED) -> None:
    """Export every certified case under `stage1_root` into per-task JSONL files."""
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: {split: [] for split in ("train", "validation", "test")}
        for task in ("observation", "evidence_selector", "evidence_writer", "pure_sft")
    }
    report = Counter()
    evidence_counts = Counter()

    for path in sorted(stage1_root.glob("case_*/stage1_run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        report["discovered"] += 1
        if run.get("schema_version") != STAGE1_SCHEMA_VERSION:
            report["skipped_incompatible_observation_contract"] += 1
            continue
        if run.get("status") not in {"certified", "validated_no_reverse"}:
            report["skipped_not_validated"] += 1
            continue
        case_id = run["case_id"]
        split = _split(case_id, seed)
        profile = run.get("model_input")
        if not profile:
            report["skipped_missing_model_input"] += 1
            continue
        observation = _observation_target(run["observation"])
        evidence_selection = run["evidence_selection"]
        evidence = run.get("evidence")
        if not evidence:
            report["skipped_missing_case_evidence"] += 1
            continue
        evidence_keys = [item["evidence_key"] for item in evidence["items"]]
        generation_evidence_bank = run.get("generation_evidence_bank")
        if not generation_evidence_bank:
            report["skipped_missing_generation_evidence_bank"] += 1
            continue
        recommendation = run["recommendation"]
        if _reverse_review_ran(run.get("reverse_review")):
            report["with_reverse_review"] += 1
        else:
            report["without_reverse_review"] += 1
        grounding = {
            section: run["recommendation_draft"][f"{section}_grounding"]
            for section in ("section_1", "section_2", "section_3", "section_4")
        }
        shared = {
            "schema_version": "stage2-export-1.0",
            "case_id": case_id,
            "split": split,
            "source_stage1_artifact": str(path),
        }
        tasks["observation"][split].append(
            _task_row(
                shared=shared,
                task="predict_case_observation",
                input_payload=profile,
                target=observation,
                audit={"observation_anchors": run["observation"]["anchors"]},
            )
        )
        tasks["evidence_selector"][split].append(
            _task_row(
                shared=shared,
                task="select_case_evidence_from_observation",
                input_payload={
                    **profile,
                    "observation": observation,
                    "generation_evidence_bank": generation_evidence_bank,
                },
                target=evidence,
                audit={"selection_contract": evidence_selection},
            )
        )
        tasks["evidence_writer"][split].append(
            _task_row(
                shared=shared,
                task="write_four_section_recommendation_from_evidence",
                input_payload={
                    **profile,
                    "observation": observation,
                    "evidence": evidence,
                },
                target=recommendation,
                audit={"section_grounding": grounding},
            )
        )
        tasks["pure_sft"][split].append(
            _task_row(
                shared=shared,
                task="write_four_section_recommendation_from_profile",
                input_payload=profile,
                target=recommendation,
                audit={"section_grounding": grounding},
            )
        )
        evidence_counts.update(evidence_keys)
        report["written"] += 1

    for task, splits in tasks.items():
        for split, rows in splits.items():
            _write_jsonl(out / task / f"{split}.jsonl", rows)
    manifest = {
        "schema_version": "stage2-export-1.0",
        "comparison": {
            "pure_sft": "P -> R",
            "observation": "P -> O",
            "evidence_selector": "P,O,full_generation_bank -> E",
            "evidence_writer": "P,O,E -> R",
        },
        "same_case_split_for_all_tasks": True,
        "prompt_version": STAGE2_PROMPT_VERSION,
        "prompt_catalog": stage2_prompt_catalog(),
        "counts": dict(report),
        "split_counts": {
            task: {split: len(rows) for split, rows in splits.items()}
            for task, splits in tasks.items()
        },
        "evidence_counts": dict(evidence_counts.most_common()),
        "all_stage1_sources_reverse_reviewed": (
            report["written"] > 0
            and report["with_reverse_review"] == report["written"]
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.stage1_dir, args.out)


if __name__ == "__main__":
    main()

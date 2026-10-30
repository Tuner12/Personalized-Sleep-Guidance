#!/usr/bin/env python3
"""Stage 1: construct the evidence-grounded dataset from unannotated wearable records (paper Section 3.1).

For every case, the Evidence Agent writes an observation O and selects evidence identifiers E,
the cards C are looked up from the Evidence Bank, the Composer Agent writes the four-section
guidance R with numeric anchors, deterministic checks verify the anchors, and the Reflection
Agent independently recovers the required evidence from R. A case is retained when the
recovered evidence matches E; otherwise the reverse-review feedback drives a revision, for at
most five cycles. Every agent call is a fresh `codex exec --ephemeral` invocation
(configs/providers.yaml), so no role sees another role's context.

    python scripts/construct_dataset.py --profiles-dir outputs/profiles --output-dir outputs/stage1
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sleepguide.stage1 import OBSERVATION_CONTRACT_ID, STAGE1_SCHEMA_VERSION
from sleepguide.stage1.construct import run as run_case

def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", type=Path, required=True,
                        help="Profile Agent output: one <case_id>.json per case")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=1500)
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional comma-separated pilot cases instead of the positional slice.",
    )
    parser.add_argument("--case-workers", type=int, default=20)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--max-stage-retries", type=int, default=3)
    parser.add_argument("--max-case-retries", type=int, default=2,
                        help="re-run a case after an error; resumes its unfinished cycle")
    parser.add_argument("--skip-reverse", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
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
    parser.add_argument("--providers", type=Path, default=Path("configs/providers.yaml"))
    parser.add_argument("--provider", default=None)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _case_namespace(args: argparse.Namespace, case_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        profiles_dir=args.profiles_dir,
        case_id=case_id,
        output_dir=args.output_dir,
        providers=args.providers,
        provider=args.provider,
        max_cycles=args.max_cycles,
        max_stage_retries=args.max_stage_retries,
        skip_reverse=args.skip_reverse,
        generation_cards=args.generation_cards,
        source_bank=args.source_bank,
    )


def _save_result(args: argparse.Namespace, case_id: str, result: dict[str, Any]) -> None:
    case_dir = args.output_dir / case_id
    _atomic_json(case_dir / "stage1_run.json", result)
    if result.get("status") in {"certified", "validated_no_reverse"}:
        _atomic_json(case_dir / "recommendation.json", result["recommendation"])


def _existing_status(args: argparse.Namespace, case_id: str) -> str | None:
    path = args.output_dir / case_id / "stage1_run.json"
    if not path.exists():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("schema_version") != STAGE1_SCHEMA_VERSION:
            return None
        return artifact.get("status")
    except json.JSONDecodeError:
        return None


def main() -> int:
    args = _args()
    if args.case_workers < 1:
        raise SystemExit("--case-workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = sorted(path.stem for path in args.profiles_dir.glob("*.json"))
    case_ids = (
        [item.strip() for item in args.case_ids.split(",") if item.strip()]
        if args.case_ids
        else available[args.start_index : args.end_index]
    )
    unknown = sorted(set(case_ids) - set(available))
    if unknown:
        raise SystemExit(f"unknown case ids: {unknown}")
    completed_statuses = {"certified", "validated_no_reverse"} if args.skip_reverse else {"certified"}
    existing_statuses = {
        case_id: _existing_status(args, case_id) for case_id in case_ids
    }
    pending = [
        case_id for case_id in case_ids if existing_statuses[case_id] not in completed_statuses
    ]
    progress_lock = threading.Lock()
    counters = {
        "requested": len(case_ids),
        "already_completed": len(case_ids) - len(pending),
        "certified": sum(status == "certified" for status in existing_statuses.values()),
        "validated_no_reverse": (
            sum(status == "validated_no_reverse" for status in existing_statuses.values())
            if args.skip_reverse
            else 0
        ),
        "failed": 0,
    }
    started = time.monotonic()

    def process(case_id: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for retry in range(args.max_case_retries + 1):
            try:
                result = run_case(_case_namespace(args, case_id))
                if result.get("status") != "failed":
                    _save_result(args, case_id, result)
                    return result
                last_error = RuntimeError(
                    str(result.get("terminal_issue") or "Stage 1 returned failed")
                )
                _save_result(args, case_id, result)
                _atomic_json(
                    args.output_dir / case_id / "case_retry.json",
                    {
                        "case_id": case_id,
                        "retry": retry,
                        "error": str(last_error),
                        "updated_at": time.time(),
                    },
                )
            except Exception as error:  # retain the case and continue the pool
                last_error = error
                _atomic_json(
                    args.output_dir / case_id / "case_retry.json",
                    {
                        "case_id": case_id,
                        "retry": retry,
                        "error": str(error),
                        "updated_at": time.time(),
                    },
                )
        result = {
            "schema_version": STAGE1_SCHEMA_VERSION,
            "observation_contract": OBSERVATION_CONTRACT_ID,
            "status": "failed",
            "case_id": case_id,
            "terminal_issue": f"case retry budget exhausted: {last_error}",
        }
        _save_result(args, case_id, result)
        return result

    def record(case_id: str, result: dict[str, Any]) -> None:
        with progress_lock:
            counters[result.get("status", "failed")] += 1
            completed = (
                counters["certified"]
                + counters["validated_no_reverse"]
                + counters["failed"]
            )
            with (args.output_dir / "progress.tsv").open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{time.time():.3f}\t{case_id}\t{result.get('status')}\t"
                    f"{completed}\t{len(case_ids)}\n"
                )
            _atomic_json(
                args.output_dir / "batch_summary.json",
                {
                    **counters,
                    "completed": completed,
                    "pending": len(case_ids) - completed,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )

    _atomic_json(
        args.output_dir / "batch_summary.json",
        {
            **counters,
            "completed": counters["certified"] + counters["validated_no_reverse"],
            "pending": len(pending),
            "elapsed_seconds": 0.0,
        },
    )
    if pending and not args.skip_warmup:
        first = pending.pop(0)
        record(first, process(first))
    with ThreadPoolExecutor(max_workers=args.case_workers) as executor:
        futures = {executor.submit(process, case_id): case_id for case_id in pending}
        for future in as_completed(futures):
            case_id = futures[future]
            record(case_id, future.result())
    summary = json.loads((args.output_dir / "batch_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

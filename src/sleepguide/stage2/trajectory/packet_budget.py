"""Step 2 of 6: fit each profile packet P within the 32,768-token context.

The Stage-1 export renders the full packet, which exceeds the context limit for many cases,
and training never truncates. The packet is trimmed by a keep-set derived from the training
split only: a metric survives with all five longitudinal views if Stage 1 ever grounds a
numeric anchor in it or names it in an observation's `relevant_metric_ids`; every other metric
keeps only its whole-record summary. After writing, every anchor in every split is re-checked
against the rendered text, and the step fails rather than emit a record whose anchor is not
visible to the model.

    python -m sleepguide.stage2.trajectory.packet_budget --source <export> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SECTIONS = ("section_1", "section_2", "section_3", "section_4")
SPLITS = ("train", "validation", "test")
# the five-view expansion lives under these; a metric outside the keep-set keeps only the
# whole-record summary, which is where the population-level facts are
VIEW_KEYS = ("rolling_7_night_trajectory", "recent_7_night_comparison",
             "latest_night_context", "fluctuation_points")


def load(source: Path, task: str, split: str) -> list[dict]:
    path = source / task / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def keep_set(train_rows: dict[str, list[dict]]) -> set[str]:
    """Metrics Stage 1 actually uses, from the training split only."""
    keep: set[str] = set()
    for row in train_rows["pure_sft"]:
        for section in SECTIONS:
            grounding = (row["audit"].get("section_grounding") or {}).get(section) or {}
            for anchor in grounding.get("case_anchors") or []:
                keep.add(anchor["metric_id"])
    for row in train_rows["observation"]:
        keep.update(row["target"].get("relevant_metric_ids") or [])
    return keep


def trim(payload: dict, keep: set[str]) -> dict:
    """Drop the five-view expansion for metrics nothing ever cites."""
    out = json.loads(json.dumps(payload))
    insight = out.get("longitudinal_insight") or {}
    for key in VIEW_KEYS:
        entries = insight.get(key)
        if isinstance(entries, list):
            insight[key] = [e for e in entries
                            if not isinstance(e, dict) or e.get("metric_id") in keep]
    profile = out.get("profile") or {}
    for key, entries in list(profile.items()):
        if key in ("metric_coverage", "weekday_weekend_comparisons") \
                and isinstance(entries, list):
            profile[key] = [e for e in entries
                            if not isinstance(e, dict) or e.get("metric_id") in keep]
    return out


def render(instruction: str, payload: dict) -> str:
    """Render the payload as compact JSON (no indentation).

    This is lossless and is what brings every case within the 32,768-token limit. Values are
    not rounded, because numeric anchors are matched exactly.
    """

    return "\n\n".join([
        instruction.strip(),
        "Input payload (JSON):",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "Return only the JSON object required by the supplied output schema.",
    ])


TASKS = ("pure_sft", "evidence_writer", "observation", "evidence_selector")


def build(source: Path, out: Path) -> None:
    """Fit every exported record's profile packet within the context budget."""

    data = {task: {split: load(source, task, split) for split in SPLITS} for task in TASKS}
    keep = keep_set({t: data[t]["train"] for t in ("pure_sft", "observation")})
    print(f"keep-set derived from the training split: {len(keep)} metrics")
    print("  " + ", ".join(sorted(keep)))

    unverified: collections.Counter = collections.Counter()
    written = 0
    for task, splits in data.items():
        for split, rows in splits.items():
            out_path = out / task / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    payload = trim(row["input"], keep)
                    text = render(row["instruction"], payload)
                    # an anchor whose metric is not in the text the model reads is a defect
                    for section in SECTIONS:
                        grounding = (row["audit"].get("section_grounding") or {}).get(section) or {}
                        for anchor in grounding.get("case_anchors") or []:
                            if f'"{anchor["metric_id"]}"' not in text:
                                unverified[anchor["metric_id"]] += 1
                    handle.write(json.dumps({
                        "case_id": row["case_id"],
                        "task": row["task"],
                        "split": split,
                        "input": text,
                        "target": json.dumps(row["target"], ensure_ascii=False, indent=2),
                        "auxiliary": row["audit"],
                    }, ensure_ascii=False) + "\n")
                    written += 1

    if unverified:
        print("\nFAILED: anchors whose metric is absent from the rendered input")
        for name, count in unverified.most_common():
            print(f"  {name:34s} {count}")
        sys.exit(1)
    print(f"\nwrote {written} records to {out}; every anchor's metric is visible")


def main() -> None:
    ap = argparse.ArgumentParser(description=build.__doc__)
    ap.add_argument("--source", type=Path, required=True, help="output of export.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()

"""Attach the reference guidance to a generation file and score it.

    python -m sleepguide.evaluation.evaluate --generated outputs/generations/<run>.jsonl \
        --test-file outputs/sft/trajectory/test.jsonl --stage1-dir outputs/stage1 \
        --output outputs/scores/<run>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sleepguide.evaluation import metrics


def attach_references(generated: Path, test_file: Path) -> int:
    """Add each case's reference sections to the generation file, in place; drop duplicates."""
    reference = {}
    for line in test_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            reference[row["case_id"]] = row["target"]
    rows = [json.loads(line) for line in
            generated.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, kept = set(), []
    for row in rows:
        if row["case_id"] in seen or row["case_id"] not in reference:
            continue
        seen.add(row["case_id"])
        target = reference[row["case_id"]]
        if target.startswith("<OBSERVATION>"):
            target = target.split("<SECTIONS>")[1].split("</SECTIONS>")[0].strip()
        row["reference"], row["run_id"] = target, None
        kept.append(row)
    with generated.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--test-file", type=Path, required=True)
    ap.add_argument("--stage1-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()

    count = attach_references(args.generated, args.test_file)
    print(f"attached references to {count} cases", flush=True)
    metrics.main(["--decoded", str(args.generated), "--test-file", str(args.test_file),
                  "--batch-root", str(args.stage1_dir), "--output", str(args.output)]
                 + (["--no-bertscore"] if args.no_bertscore else []))


if __name__ == "__main__":
    main()

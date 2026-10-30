"""Step 4 of 6: list the evidence-card thresholds each case crosses, ahead of the payload.

Eight of the 22 cards state a threshold (metric, comparator, value). Whether a case crosses it
is arithmetic on P, so it is computed deterministically and listed in the input, and the
model only decides which crossed reference is worth citing. Every crossed card is listed,
including ones Stage 1 rarely selects, so the list is a computation, not a shortlist of
the answer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sleepguide.stage2.trajectory.anchors import packet_of  # noqa: E402
from sleepguide.stage2.trajectory.card_conditions import bank, condition_holds, gated_cards  # noqa: E402

HEADING = ("Reference thresholds this record crosses, computed from the packet. Crossing a "
           "threshold makes a reference applicable, not necessarily worth citing.")


def block(packet: dict) -> str:
    crossed = sorted(key for key in gated_cards() if condition_holds(packet, key))
    if not crossed:
        return HEADING + "\n  (none)"
    lines = []
    for key in crossed:
        entry = bank()[key]
        reference = entry["reference_values"][0]
        lines.append(f"  {key}  ({reference['metric_id']} "
                     f"{reference['comparator']} {reference['value']} {reference['unit']})")
    return HEADING + "\n" + "\n".join(lines)


def build(source: Path, out: Path) -> None:
    """List, ahead of the payload, the evidence-card thresholds each case crosses."""

    marker = "Input payload (JSON):"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = [json.loads(line) for line in
                (source / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        sizes, records = [], []
        for row in rows:
            packet = packet_of(row["input"])
            text = block(packet)
            sizes.append(text.count("\n  ") if "(none)" not in text else 0)
            head, tail = row["input"].split(marker, 1)
            # Ahead of the payload, so the model reads the computed result before the raw
            # numbers it would otherwise have to derive it from.
            new = dict(row)
            new["input"] = head + text + "\n\n" + marker + tail
            assert new["target"] == row["target"], row["case_id"]
            assert marker in new["input"] and HEADING in new["input"], row["case_id"]
            records.append(new)

        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}  {len(records)} rows  mean crossed {statistics.mean(sizes):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()

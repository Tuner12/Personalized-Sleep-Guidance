"""Step 6 of 6: final trajectory targets -- conditional verdict clauses and timing comparisons.

Two edits to the canonical observation from step 5:

* a reference-crossing verdict clause is kept only when a card carrying that clause is among
  the case's selected evidence, matching how Stage 1 mentions an axis only when its card
  applies;
* when a timing card is selected and O has no clock sentence, the packet's own clock
  comparisons (sleep onset, final wake, midpoint, weekday/weekend) are appended with the
  same renderer.

Both are computable at inference from P; E, C and R are unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sleepguide.stage2.trajectory.anchors import packet_of, resolve  # noqa: E402
from sleepguide.stage2.trajectory.observation_render import REFERENCE_CLAUSE, sentences  # noqa: E402
from sleepguide.stage2.trajectory.units import unit_for  # noqa: E402

TIMING_CARDS = {"make-sleep-timing-more-consistent",
                "sleep-timing-alone-does-not-confirm-circadian-phase",
                "unwanted-sleep-timing-may-need-clinical-review"}
CLOCK_METRICS = ("sleep_onset_clock_min", "final_wake_clock_min", "sleep_midpoint_clock_min")
# Any surface the renderer gives a clock metric: the center phrases ("your sleep usually
# started about 9:58 pm") and the delta/gap phrases built from the metric's short name
# ("Your sleep start was later by..."). Missing the second family would add
# timing twice on every case whose anchors carried a weekpart gap without centers.
CLOCK_HEADS = ("your sleep usually started", "you finally woke", "the midpoint of sleep fell",
               "sleep start", "final waking", "sleep midpoint")

# clause text -> the cards that carry it (the two long-wake cards share one text)
CLAUSE_CARDS: dict[str, set[str]] = {}
for card, clause in REFERENCE_CLAUSE.items():
    CLAUSE_CARDS.setdefault(clause, set()).add(card)


def timing_block(record: dict) -> str:
    """The packet's clock comparisons, rendered canonically, clock metrics only."""
    packet = packet_of(record["input"])
    stats: dict[str, dict[str, tuple[float, str | None]]] = {}

    def put(metric: str, statistic: str, value) -> None:
        if value is not None:
            stats.setdefault(metric, {})[statistic] = (
                float(value), unit_for(metric, statistic))

    insight = packet.get("longitudinal_insight") or {}
    for entry in insight.get("recent_7_night_comparison") or []:
        metric = entry.get("metric_id")
        if metric in CLOCK_METRICS:
            for statistic in ("recent_7_center", "previous_7_center"):
                put(metric, statistic, resolve(packet, metric, "insight", statistic))
    for entry in (packet.get("profile") or {}).get("weekday_weekend_comparisons") or []:
        metric = entry.get("metric_id")
        if metric in CLOCK_METRICS and entry.get("reportable"):
            put(metric, "weekday_center", entry.get("weekday_center"))
            put(metric, "weekend_center", entry.get("weekend_center"))
            put(metric, "weekend_minus_weekday", entry.get("weekend_minus_weekday"))
    # One line per clock metric, so the observation stays short: the weekday/weekend contrast
    # where the packet reports one, else the recent-vs-previous comparison. No medians.
    for metric in CLOCK_METRICS:
        s = stats.get(metric)
        if s and "weekday_center" in s:
            s.pop("recent_7_center", None)
            s.pop("previous_7_center", None)
    if not stats:
        return ""
    rendered = sentences(stats, {})  # empty packet: no reference clauses re-fire
    # drop any clause line, so the selected-card filter alone decides which clauses appear
    return "\n".join(line for line in rendered.splitlines()
                     if line not in CLAUSE_CARDS)


def rework(row: dict, tally: dict) -> str:
    target = row["target"]
    obs = target.split("<OBSERVATION>")[1].split("</OBSERVATION>")[0].strip("\n")
    gold = {s["dossier_id"] for s in row["auxiliary"]["evidence_selection"]}

    kept = []
    for line in obs.splitlines():
        cards = CLAUSE_CARDS.get(line)
        if cards is not None and not (cards & gold):
            tally["clauses_dropped"] += 1
            continue
        if cards is not None:
            tally["clauses_kept"] += 1
        kept.append(line)

    if gold & TIMING_CARDS and not any(h in obs for h in CLOCK_HEADS):
        block = timing_block(row)
        if block:
            tally["timing_blocks_added"] += 1
            tally["timing_lines_added"] += len(block.splitlines())
            # clock sentences belong with the metric sentences, ahead of any kept clause
            clause_at = next((i for i, line in enumerate(kept) if line in CLAUSE_CARDS),
                             len(kept))
            kept[clause_at:clause_at] = block.splitlines()
        else:
            tally["timing_gold_no_packet_clock"] += 1

    new_obs = "\n".join(kept)
    head, tail = target.split("<OBSERVATION>", 1)
    return head + "<OBSERVATION>\n" + new_obs + "\n</OBSERVATION>" \
        + tail.split("</OBSERVATION>", 1)[1]


def build(source: Path, out: Path) -> None:
    """Keep verdict clauses conditional on the selected cards and add timing comparisons."""
    out.mkdir(parents=True, exist_ok=True)
    tally = {"clauses_dropped": 0, "clauses_kept": 0, "timing_blocks_added": 0,
             "timing_lines_added": 0, "timing_gold_no_packet_clock": 0}
    for split in ("train", "test"):
        before, after = [], []
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for line in (source / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                before.append(len(row["target"].split("<OBSERVATION>")[1]
                                  .split("</OBSERVATION>")[0].strip("\n").splitlines()))
                row["target"] = rework(row, tally)
                after.append(len(row["target"].split("<OBSERVATION>")[1]
                                 .split("</OBSERVATION>")[0].strip("\n").splitlines()))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: mean O lines {statistics.mean(before):.2f} -> "
              f"{statistics.mean(after):.2f}  (Stage-1 reference 2.75)")
    print(tally)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()

"""Audit canonical observations anchor by anchor before training.

Every certified anchor must be rendered, and rendered with the right value; the build is
rejected on any missing or wrong anchor.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from sleepguide.stage2.trajectory.observation_render import from_anchors  # noqa: E402
from sleepguide.stage2.trajectory.value_format import render  # noqa: E402

CLOCK_METRICS = {"sleep_onset_clock_min", "final_wake_clock_min", "sleep_midpoint_clock_min"}


def expected_surface(anchor: dict) -> tuple[str | None, str]:
    value = float(anchor["value"])
    unit = anchor.get("unit")
    if "minus" in (anchor.get("statistic") or ""):
        value = abs(value)  # differences are stated as "<direction> by about <magnitude>"
    kind = {"%": "percent", "clock_minute": "clock", "min": "duration",
            "count/night": "count", "night": "nights"}.get(unit, "other")
    return render(value, unit), kind


def present(text: str, surface: str | None, kind: str) -> bool:
    if surface is None:
        return False
    pattern = re.escape(surface)
    if kind == "duration":
        # "50 minutes" must not be the tail of "1 hour 50 minutes", and "1 hour" must not
        # be the head of "1 hour 10 minutes".
        for match in re.finditer(pattern, text):
            if re.search(r"hours?\s+$", text[:match.start()]):
                continue
            if re.match(r"\s+\d+\s+minutes?", text[match.end():]):
                continue
            return True
        return False
    if kind == "count":
        return re.search(r"about " + pattern + r"\b(?!\s*(hours?|minutes?|nights?|%|:))",
                         text) is not None
    if kind == "nights":
        return re.search(r"about " + pattern + r"\s+nights", text) is not None
    return surface in text


def buggy_variant(anchor: dict, text: str) -> str | None:
    value = float(anchor["value"])
    unit = anchor.get("unit")
    if unit == "clock_minute":
        wrong = render(value, "min")
        if wrong and present(text, wrong, "duration"):
            return f'clock value rendered as duration "{wrong}"'
    if unit == "count/night":
        wrong = render(abs(value), "min")
        if wrong and re.search(r"about " + re.escape(wrong) + r" wake episodes", text):
            return f'count rendered as duration "{wrong}"'
    if unit == "min" and anchor["metric_id"] in CLOCK_METRICS \
            and "minus" not in (anchor.get("statistic") or ""):
        wrong = render(value, "clock_minute")
        if wrong and wrong in text:
            return f'duration rendered as clock "{wrong}"'
    return None


def observation_of(row: dict, use: str) -> str:
    if use == "anchors":
        return from_anchors(row)
    return row["target"].split("<OBSERVATION>")[1].split("</OBSERVATION>")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="arm directory holding train.jsonl/test.jsonl")
    ap.add_argument("--use", choices=("anchors", "target"), required=True)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()

    totals = collections.Counter()
    by_statistic = collections.Counter()
    examples: list[str] = []
    for split in args.splits.split(","):
        for line in (args.source / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            anchors = [a for a in row.get("auxiliary", {}).get("certified_anchors") or []
                       if a.get("origin") != "daily_observation"]
            if not anchors:
                totals["cases_without_anchors"] += 1
                continue
            text = observation_of(row, args.use)
            seen = set()
            for anchor in anchors:
                key = (anchor["metric_id"], anchor.get("origin"), anchor.get("statistic"),
                       float(anchor["value"]))
                if key in seen:
                    continue
                seen.add(key)
                surface, kind = expected_surface(anchor)
                if present(text, surface, kind):
                    totals["ok"] += 1
                    continue
                bug = buggy_variant(anchor, text)
                status = "wrong_render" if bug else "missing"
                totals[status] += 1
                by_statistic[(status, anchor.get("statistic"))] += 1
                if len(examples) < 12:
                    examples.append(f'{row["case_id"]} {split} {status}: '
                                    f'{anchor["metric_id"]}.{anchor.get("origin")}.'
                                    f'{anchor.get("statistic")} = {anchor["value"]} '
                                    f'{anchor.get("unit")} -> '
                                    f'{bug or f"expected {surface!r} ({kind})"}')

    audited = totals["ok"] + totals["missing"] + totals["wrong_render"]
    print(f"audited {audited} unique anchors: ok {totals['ok']}  "
          f"missing {totals['missing']}  wrong {totals['wrong_render']}  "
          f"(cases without anchors: {totals['cases_without_anchors']})")
    for (status, statistic), n in sorted(by_statistic.items(), key=lambda kv: -kv[1]):
        print(f"  {status:12s} {statistic:30s} {n}")
    for example in examples:
        print(" ", example)
    sys.exit(1 if totals["missing"] or totals["wrong_render"] else 0)


if __name__ == "__main__":
    main()

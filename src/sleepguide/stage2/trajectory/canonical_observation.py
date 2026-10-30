"""Step 5 of 6: rewrite each target observation O into canonical per-axis sentences.

Stage-1 observations vary freely in wording, which a small model cannot reproduce. The
canonical form states the same certified anchors (metric, window, value) with a small set of
fixed sentence shapes, so what remains to learn is which axes matter. Everything after
`</OBSERVATION>` stays byte-identical; a case with no usable anchors keeps its original O.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sleepguide.stage2.trajectory.observation_render import from_anchors  # noqa: E402

OLD_OBSERVATION = """`<OBSERVATION>` the decision-relevant facts of this record, in plain sentences. State the
value of every quantity the Pattern Analysis and Personalized Recommendation will rest on. When a quantity changed
between windows, give the change itself and not only the two endpoints. When sleep timing
matters, give the clock times -- when sleep began, when it ended, where its midpoint fell --
and when weekday and weekend differ, give both. `</OBSERVATION>`"""

NEW_OBSERVATION = """`<OBSERVATION>` the decision-relevant quantities, one sentence per comparison: the recent
value against the preceding window with the direction of change, the whole-record value,
weekday against weekend where the record supports it, and any quality reference the record
crosses, restated from the threshold list above. `</OBSERVATION>`"""


def build(source: Path, out: Path) -> None:
    """Rewrite each observation O into canonical per-axis sentences."""

    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = [json.loads(line) for line in
                (source / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        records, lengths = [], []
        for row in rows:
            canonical = from_anchors(row)
            if not canonical.strip():
                # A case with no usable anchors keeps its original observation; the segment
                # must never be empty, and a small number of training cases have no anchors at all.
                canonical = (row["target"].split("<OBSERVATION>")[1]
                             .split("</OBSERVATION>")[0].strip())
            lengths.append(len(canonical.splitlines()))
            head, rest = row["target"].split("<OBSERVATION>", 1)
            _, tail = rest.split("</OBSERVATION>", 1)
            new = dict(row)
            new["target"] = (head + "<OBSERVATION>\n" + canonical + "\n</OBSERVATION>" + tail)

            assert OLD_OBSERVATION in new["input"], row["case_id"]
            new["input"] = new["input"].replace(OLD_OBSERVATION, NEW_OBSERVATION, 1)

            # Everything downstream of the observation must be byte-identical.
            assert new["target"].split("</OBSERVATION>", 1)[1] == tail, row["case_id"]
            for tag in ("EVIDENCE", "CARDS", "SECTIONS"):
                before = row["target"].split(f"<{tag}>")[1].split(f"</{tag}>")[0]
                after = new["target"].split(f"<{tag}>")[1].split(f"</{tag}>")[0]
                assert before == after, f"{row['case_id']}: {tag} changed"
            records.append(new)

        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}  {len(records)} rows  observation mean {statistics.mean(lengths):.1f} lines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()

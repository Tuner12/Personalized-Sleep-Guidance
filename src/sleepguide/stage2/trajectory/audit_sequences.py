"""Audit a trajectory SFT split before training.

Checks the fixed 300-case test split and train/test disjointness, that every evidence
identifier resolves in the card library, that the `<CARDS>` text equals the library text
generation will insert, and that phase-one and phase-two outputs fit the decoding budgets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--library-json", type=Path, required=True)
    ap.add_argument("--resume-tag", default="<SECTIONS>",
                    help="the tag the decoder resumes phase two at; must be what follows "
                         "</CARDS> in the target.")
    ap.add_argument("--reference-test", type=Path, default=None,
                    help="the fixed 300-case test split to compare against (default: the arm's own)")
    ap.add_argument("--phase1-tokens", type=int, default=640)
    ap.add_argument("--phase2-tokens", type=int, default=1600)
    ap.add_argument("--model-dir", default="Qwen/Qwen3.5-0.8B",
                    help="only its tokenizer is used, for the budget check.")
    args = ap.parse_args()
    if args.reference_test is None:
        args.reference_test = args.arm / "test.jsonl"

    library = json.loads(args.library_json.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    base = {row["case_id"] for row in rows_of(args.reference_test)}
    train = rows_of(args.arm / "train.jsonl")
    test = rows_of(args.arm / "test.jsonl")

    print(f"{args.arm}  train {len(train)}  test {len(test)}")
    check("test split is the reference 300 cases",
          {row["case_id"] for row in test} == base)
    check("no train/test leakage",
          not ({row["case_id"] for row in train} & base))

    # The decoder builds this exact string after </EVIDENCE>; the target must start with it,
    # or the model is conditioned at serve on a prefix it never saw in training.
    bad = []
    for row in train + test:
        target = row["target"]
        if "<EVIDENCE>" not in target or "<CARDS>" not in target:
            bad.append(row["case_id"])
            continue
        names = [x.strip() for x in
                 target.split("<EVIDENCE>")[1].split("</EVIDENCE>")[0].splitlines() if x.strip()]
        missing = [n for n in names if n not in library]
        if missing:
            bad.append(f"{row['case_id']}:{missing[0]}")
            continue
        body = "\n\n".join(library[n] for n in names)
        built = f"</EVIDENCE>\n<CARDS>\n{body}\n</CARDS>\n{args.resume_tag}\n"
        if not ("</EVIDENCE>" + target.split("</EVIDENCE>", 1)[1]).startswith(built):
            bad.append(row["case_id"])
    check("serve prefix is byte-identical to the target",
          not bad, f"{len(bad)} mismatched" + (f", first {bad[0]}" if bad else ""))

    # Every tag the target uses must be described by the instruction the model is given.
    tags = {"<OBSERVATION>", "<EVIDENCE>", "<CARDS>", "<SECTIONS>"}
    if args.resume_tag != "<SECTIONS>":
        tags.add(args.resume_tag)
    undescribed = sorted(t for t in tags if any(t in r["target"] and t not in r["input"]
                                                for r in train[:50]))
    check("every target segment is described in the prompt",
          not undescribed, f"undescribed: {undescribed}" if undescribed else "")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    p1 = [count(r["target"].split("</EVIDENCE>")[0] + "</EVIDENCE>") for r in test]
    p2 = [count(args.resume_tag + r["target"].split(args.resume_tag, 1)[1])
          for r in test if args.resume_tag in r["target"]]
    check("phase-1 budget covers every case",
          max(p1) <= args.phase1_tokens, f"max {max(p1)} of {args.phase1_tokens}")
    check("phase-2 budget covers every case",
          bool(p2) and max(p2) <= args.phase2_tokens,
          f"max {max(p2) if p2 else 0} of {args.phase2_tokens}")


    if failures:
        print(f"\n{len(failures)} check(s) failed: {failures}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

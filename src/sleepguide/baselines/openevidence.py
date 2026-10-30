"""OpenEvidence (Osler) baseline, run by hand because OpenEvidence has no API.

    python -m sleepguide.baselines.openevidence pack    --out outputs/openevidence
    # paste each prompts/<case_id>.txt into a new OpenEvidence conversation, unchanged, and
    # save the first complete reply as responses/<case_id>.txt
    python -m sleepguide.baselines.openevidence collect --pack outputs/openevidence \
        --out outputs/generations/openevidence.jsonl

Prompts are identical to the GPT-5.6-Sol prompts (Composer contract, P, all 22 cards). Cases are ordered round-robin across
reference-evidence strata, so any prefix of `order.json` is a balanced subsample.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sleepguide.baselines.llm_prompting import build_prompt, parse
from sleepguide.stage2.tokenize import TRAJECTORY_TASK, load_records


def pack(test_file: Path, card_library: Path, out: Path, seed: int = 0) -> None:
    cards = json.loads(card_library.read_text())
    records = load_records(test_file, task=TRAJECTORY_TASK)
    strata = defaultdict(list)
    for r in records:
        block = r["target"].split("<EVIDENCE>")[1].split("</EVIDENCE>")[0]
        strata[frozenset(x.strip() for x in block.splitlines() if x.strip())].append(r)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)
    order = sorted(strata, key=lambda k: -len(strata[k]))
    picked, index = [], 0
    while any(strata[k] for k in order):
        key = order[index % len(order)]
        if strata[key]:
            picked.append(strata[key].pop())
        index += 1
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    for r in picked:
        (out / "prompts" / f"{r['case_id']}.txt").write_text(build_prompt(r["input"], "gpt", cards),
                                                              encoding="utf-8")
    (out / "order.json").write_text(json.dumps([r["case_id"] for r in picked], indent=1))
    print(f"{len(picked)} prompts -> {out / 'prompts'}")


def collect(pack_dir: Path, card_library: Path, out: Path) -> None:
    valid = set(json.loads(card_library.read_text()))
    rows, missing = [], 0
    for prompt in sorted((pack_dir / "prompts").glob("case_*.txt")):
        response = pack_dir / "responses" / prompt.name
        if not response.exists():
            missing += 1
            continue
        text = response.read_text(encoding="utf-8", errors="replace")
        ids, generated = parse(text, valid)
        if "<SECTIONS>" not in text:
            # an untagged reply: take the first JSON object anywhere in it, else nothing
            try:
                generated = json.dumps(json.JSONDecoder().raw_decode(text[text.index("{"):])[0],
                                       ensure_ascii=False)
            except Exception:
                generated = ""
        rows.append({"case_id": prompt.stem, "generated": generated, "evidence_selection": ids})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} parsed -> {out}; missing {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--test-file", type=Path, default=Path("outputs/sft/trajectory/test.jsonl"))
    p.add_argument("--card-library", type=Path, default=Path("outputs/sft/card_library.json"))
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--pack", type=Path, required=True)
    c.add_argument("--card-library", type=Path, default=Path("outputs/sft/card_library.json"))
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "pack":
        pack(args.test_file, args.card_library, args.out)
    else:
        collect(args.pack, args.card_library, args.out)


if __name__ == "__main__":
    main()

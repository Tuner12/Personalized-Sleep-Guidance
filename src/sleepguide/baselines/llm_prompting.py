"""Prompting baselines: GPT-5.6-Sol and GPT Health (paper Table 1).

The baselines receive essentially the Composer Agent's prompt, without prompt optimization:
the same writing contract and record summary P that our model reads, plus the full text of all
22 evidence cards, from which they choose themselves. Each case is one single-turn
`codex exec` call with no tools and no few-shot examples, and the four sections are written in
that one call (no Reflection Agent review). ChatGPT Health has no API, so GPT Health is
GPT-5.6-Sol with `prompts/baselines/health_assistant_instruction.md` prepended.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from sleepguide.paths import PROMPTS

SYSTEMS = ("gpt", "gpt_health")
CARDS_HEADER = "Evidence cards (all cards in the bank; use only those that apply to this record):"
CARD_ID = re.compile(r"[a-z0-9][a-z0-9-]+")


def output_format() -> str:
    return "\n\n" + (PROMPTS / "baselines/output_format.md").read_text(encoding="utf-8").rstrip("\n")


def health_instruction() -> str:
    text = (PROMPTS / "baselines/health_assistant_instruction.md").read_text(encoding="utf-8")
    return text.rstrip("\n") + "\n\n"


def cards_block(cards: dict[str, str]) -> str:
    """The full text of every card, in library order."""
    return f"{CARDS_HEADER}\n<CARDS>\n" + "\n\n".join(cards.values()) + "\n</CARDS>"


def build_prompt(record_input: str, system: str, cards: dict[str, str]) -> str:
    """Record input (contract + P), all 22 cards before its closing instruction, output format."""
    body, closing = record_input.rsplit("\n\n", 1)
    prefix = health_instruction() if system == "gpt_health" else ""
    return f"{prefix}{body}\n\n{cards_block(cards)}\n\n{closing}{output_format()}"


def call_gpt(prompt: str, effort: str = "none") -> str:
    out = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
         "-m", "gpt-5.6-sol", "-c", f'model_reasoning_effort="{effort}"', prompt],
        capture_output=True, text=True, timeout=400)
    return out.stdout


def parse(text: str, valid_ids: set[str]) -> tuple[list[str], str]:
    """Card identifiers from <EVIDENCE> and the four sections from <SECTIONS>."""
    ids = []
    evidence = re.search(r"<EVIDENCE>(.*?)</EVIDENCE>", text, re.S)
    if evidence:
        for line in evidence.group(1).splitlines():
            for token in CARD_ID.findall(line.strip()):
                if token in valid_ids and token not in ids:
                    ids.append(token)
    generated = ""
    sections = re.search(r"<SECTIONS>(.*?)</SECTIONS>", text, re.S)
    if sections:
        body = sections.group(1)
        try:
            generated = json.dumps(json.JSONDecoder().raw_decode(body[body.index("{"):])[0],
                                   ensure_ascii=False)
        except Exception:
            generated = body.strip()
    return ids, generated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--test-file", type=Path, default=Path("outputs/sft/trajectory/test.jsonl"))
    ap.add_argument("--card-library", type=Path, default=Path("outputs/sft/card_library.json"))
    ap.add_argument("--out", type=Path, help="generation file (required unless --show-prompt)")
    ap.add_argument("--show-prompt", nargs="?", const="", metavar="CASE_ID",
                    help="print the full prompt for one case (default: the first) and exit")
    args = ap.parse_args()

    cards = json.loads(args.card_library.read_text())
    valid = set(cards)
    records = [json.loads(l) for l in args.test_file.read_text().splitlines() if l.strip()]
    if args.show_prompt is not None:
        record = next((r for r in records if r["case_id"] == args.show_prompt), records[0]) \
            if args.show_prompt else records[0]
        print(build_prompt(record["input"], args.system, cards))
        return
    if args.out is None:
        ap.error("--out is required unless --show-prompt is given")
    done = set()
    if args.out.exists():
        done = {json.loads(l)["case_id"] for l in args.out.read_text().splitlines() if l.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["case_id"] in done:
                continue
            try:
                text = call_gpt(build_prompt(record["input"], args.system, cards))
            except subprocess.TimeoutExpired:
                text = ""
            ids, generated = parse(text, valid)
            handle.write(json.dumps({"case_id": record["case_id"], "generated": generated,
                                     "evidence_selection": ids}, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

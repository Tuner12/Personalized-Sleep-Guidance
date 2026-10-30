"""Two-phase generation with deterministic evidence lookup (paper Eq. 4).

Phase 1 generates from P up to `</EVIDENCE>`; with `--observations` (the Best-of-N pick) the
given observation is inserted and only the evidence identifiers are generated. Identifiers are
matched exactly against the card library; any that are not in it are dropped and counted.
The cards' text is inserted verbatim as `<CARDS>`, and phase 2 generates the four-section
guidance R. The model writes identifiers, never evidence text. `--no-cards` inserts an empty
`<CARDS>` block (the retrieval ablation).

Output: one JSON line per case with `generated` (the four sections), `evidence_selection`
and `observation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleepguide.paths import EVIDENCE_BANK
from sleepguide.stage2.tokenize import TRAJECTORY_TASK, load_records

OBSERVATION_OPEN, OBSERVATION_CLOSE = "<OBSERVATION>", "</OBSERVATION>"
EVIDENCE_OPEN, EVIDENCE_CLOSE = "<EVIDENCE>", "</EVIDENCE>"


def parse_identifiers(head: str, cards: dict[str, str]) -> tuple[list[str], int]:
    """Library identifiers named in the `<EVIDENCE>` block, in order and deduplicated,
    plus the number of lines that named no library card."""

    block = head.split(EVIDENCE_OPEN, 1)[1] if EVIDENCE_OPEN in head else ""
    names, seen, invalid = [], set(), 0
    for line in block.splitlines():
        name = line.strip()
        if not name or name.startswith("<"):
            continue
        if name not in cards:
            invalid += 1
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names, invalid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--card-library", type=Path, default=EVIDENCE_BANK / "card_library.json")
    ap.add_argument("--observations", type=Path, default=None,
                    help="{case_id: observation} JSON, e.g. the Best-of-N pick")
    ap.add_argument("--no-cards", action="store_true",
                    help="ablation: insert an empty <CARDS> block instead of the cards")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--phase1-tokens", type=int, default=512)
    ap.add_argument("--phase2-tokens", type=int, default=1600)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    # Left padding, so every batched continuation starts right after its own prompt.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad = tok.pad_token_id or tok.eos_token_id

    cards = json.loads(args.card_library.read_text(encoding="utf-8"))
    given = (json.loads(args.observations.read_text(encoding="utf-8"))
             if args.observations else None)
    records = load_records(args.data_dir / f"{args.split}.jsonl", task=TRAJECTORY_TASK)
    records = records[args.start:] if args.limit is None else \
        records[args.start:args.start + args.limit]
    print(f"{len(records)} cases, library {len(cards)} cards", flush=True)

    def prompt_of(record: dict) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": record["input"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(texts: list[str], max_new: int, stop: str | None = None) -> list[str]:
        batch = tok(texts, add_special_tokens=False, return_tensors="pt",
                    padding=True).to(model.device)
        extra = {"stop_strings": [stop], "tokenizer": tok} if stop else {}
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new, num_beams=1,
                                 do_sample=False, pad_token_id=pad, **extra)
        width = batch["input_ids"].shape[1]
        return [tok.decode(row[width:], skip_special_tokens=True) for row in out]

    invalid_total = unterminated = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for offset in range(0, len(records), args.batch_size):
            chunk = records[offset:offset + args.batch_size]
            prompts = [prompt_of(record) for record in chunk]
            if given is not None:
                heads = [f"{OBSERVATION_OPEN}\n{given[r['case_id']].strip()}\n"
                         f"{OBSERVATION_CLOSE}\n{EVIDENCE_OPEN}\n" for r in chunk]
                grown = generate([p + h for p, h in zip(prompts, heads)],
                                 args.phase1_tokens, stop=EVIDENCE_CLOSE)
                phase1 = [h + g for h, g in zip(heads, grown)]
            else:
                phase1 = generate(prompts, args.phase1_tokens, stop=EVIDENCE_CLOSE)

            resumed, picks, heads = [], [], []
            for prompt, text in zip(prompts, phase1):
                if EVIDENCE_CLOSE not in text:
                    unterminated += 1
                head = text.split(EVIDENCE_CLOSE)[0]
                names, invalid = parse_identifiers(head, cards)
                invalid_total += invalid
                body = "" if args.no_cards else "\n\n".join(cards[n] for n in names) + "\n"
                resumed.append(f"{prompt}{head}{EVIDENCE_CLOSE}\n<CARDS>\n{body}</CARDS>\n<SECTIONS>\n")
                picks.append(names)
                heads.append(head)
            sections = generate(resumed, args.phase2_tokens)

            for record, names, head, text in zip(chunk, picks, heads, sections):
                observation = (head.split(OBSERVATION_OPEN)[1].split(OBSERVATION_CLOSE)[0].strip()
                               if OBSERVATION_OPEN in head and OBSERVATION_CLOSE in head else "")
                handle.write(json.dumps({
                    "case_id": record["case_id"],
                    "generated": text,
                    "evidence_selection": names,
                    "observation": observation,
                }, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"identifiers not in library: {invalid_total}")
    print(f"phase 1 did not emit {EVIDENCE_CLOSE}: {unterminated}")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()

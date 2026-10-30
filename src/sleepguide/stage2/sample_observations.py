"""Sample N observation candidates per case for Best-of-N (stops at `</OBSERVATION>`).

Writes one JSON file per sample index, `<out stem>_k{i}.json`, mapping case_id -> observation.
The paper draws two batches of 8 at temperature 0.8 (seeds 81 and 82), N = 16. With
`--num-samples 1` decoding is greedy and a single `<out>` file is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleepguide.stage2.tokenize import TRAJECTORY_TASK, load_records

OPEN, CLOSE = "<OBSERVATION>", "</OBSERVATION>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    records = load_records(args.data_dir / f"{args.split}.jsonl", task=TRAJECTORY_TASK)

    sampling = args.num_samples > 1
    if sampling:
        torch.manual_seed(args.seed)
    unterminated = 0
    outs: list[dict[str, str]] = [{} for _ in range(args.num_samples)]
    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        texts = [tok.apply_chat_template(
                     [{"role": "user", "content": record["input"]}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 + OPEN + "\n" for record in chunk]
        batch = tok(texts, add_special_tokens=False, return_tensors="pt",
                    padding=True).to(model.device)
        with torch.no_grad():
            grown = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                   do_sample=sampling,
                                   temperature=args.temperature if sampling else None,
                                   num_return_sequences=args.num_samples,
                                   num_beams=1, pad_token_id=tok.pad_token_id,
                                   stop_strings=[CLOSE], tokenizer=tok)
        width = batch["input_ids"].shape[1]
        for i, record in enumerate(chunk):
            for k in range(args.num_samples):
                text = tok.decode(grown[i * args.num_samples + k][width:],
                                  skip_special_tokens=True)
                if CLOSE not in text:
                    unterminated += 1
                outs[k][record["case_id"]] = text.split(CLOSE)[0].strip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if sampling:
        for k, out in enumerate(outs):
            path = args.out.with_name(args.out.stem + f"_k{k}" + args.out.suffix)
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        args.out.write_text(json.dumps(outs[0], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"did not emit {CLOSE}: {unterminated} -> {args.out.parent}")


if __name__ == "__main__":
    main()

"""Control data: Guidance-only SFT given the full text of all 22 evidence cards.

The input is the Guidance-only input (P -> R) with every card of the library inserted, verbatim
and in library order, before the final output instruction; the target is unchanged. The block
is identical for every case, so the model sees the evidence but gets no supervision on which
card applies. This separates evidence access from trajectory supervision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TAIL = "\n\nReturn only the JSON object required by the supplied output schema."
HEADER = "Evidence cards (all cards in the bank; use only those that apply to this record):"


def build(guidance_only: Path, card_library: Path, out: Path) -> None:
    """Write `out/{train,test}.jsonl` from the Guidance-only split."""
    library = json.loads(card_library.read_text(encoding="utf-8"))
    keys = list(library)
    block = f"\n\n{HEADER}\n<CARDS>\n" + "\n\n".join(library[k] for k in keys) + "\n</CARDS>"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = [json.loads(line) for line in
                (guidance_only / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                assert row["input"].endswith(TAIL), row["case_id"]
                text = row["input"][:-len(TAIL)] + block + TAIL
                handle.write(json.dumps({**row, "input": text, "cards_used": keys},
                                        ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--guidance-only", type=Path, required=True)
    ap.add_argument("--card-library", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.guidance_only, args.card_library, args.out)


if __name__ == "__main__":
    main()

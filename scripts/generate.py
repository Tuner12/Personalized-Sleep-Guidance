#!/usr/bin/env python3
"""Generate guidance for the held-out cases, greedily or with Best-of-N (paper Sections 3.2-3.3).

    # paper method: Best-of-16 (two draws of 8 at T=0.8, seeds 81 and 82)
    python scripts/generate.py --model-dir outputs/checkpoints/qwen3.5-0.8b/trajectory_sft_seed1 \
        --best-of-n 16 --out outputs/generations/qwen3.5-0.8b_seed1_bon16.jsonl

    # greedy ablation
    python scripts/generate.py --model-dir ... --out outputs/generations/..._greedy.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DRAW_SIZE, DRAW_SEEDS = 8, (81, 82)


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", f"sleepguide.stage2.{module}", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("outputs/sft/trajectory"))
    ap.add_argument("--card-library", type=Path, default=Path("outputs/sft/card_library.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--best-of-n", type=int, default=1, choices=(1, 16),
                    help="1 = greedy; 16 = the paper's Best-of-N")
    ap.add_argument("--no-cards", action="store_true", help="retrieval ablation")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    common = ["--model-dir", str(args.model_dir), "--data-dir", str(args.data_dir)]
    extra = ["--no-cards"] if args.no_cards else []
    if args.best_of_n > 1:
        work = args.out.with_suffix("")
        draws = []
        for index, seed in enumerate(DRAW_SEEDS):
            prefix = work / f"draw{index}"
            run("sample_observations", *common, "--num-samples", str(DRAW_SIZE),
                "--temperature", "0.8", "--seed", str(seed), "--out", f"{prefix}.json")
            draws.append(str(prefix))
        picked = work / "picked_observations.json"
        run("best_of_n", "--train-file", str(args.data_dir / "train.jsonl"),
            "--test-file", str(args.data_dir / "test.jsonl"), "--samples", *draws,
            "--per-prefix", str(DRAW_SIZE), "--out", str(picked))
        extra += ["--observations", str(picked)]
    run("generate", *common, "--card-library", str(args.card_library),
        "--batch-size", str(args.batch_size), "--out", str(args.out), *extra)


if __name__ == "__main__":
    main()

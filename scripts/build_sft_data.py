#!/usr/bin/env python3
"""Build the Stage-2 SFT data from a Stage-1 release (paper Section 3.2).

    python scripts/build_sft_data.py --stage1-dir outputs/stage1 --out outputs/sft

Six deterministic steps, each writing its own directory under `<out>/steps/`:

    export                 Stage-1 cases -> per-task records          trajectory/export.py
    packet_budget          fit P within 32,768 tokens                 trajectory/packet_budget.py
    sequences              Guidance-only and [P;O;E;C;R] sequences,   trajectory/sequences.py
                             1,200/300 split, card library
    threshold_flags        list crossed card thresholds in P          trajectory/threshold_flags.py
    canonical_observation  canonical per-axis observation O           trajectory/canonical_observation.py
    observation_verdicts   conditional verdict clauses + timing       trajectory/observation_verdicts.py

Final outputs:

    <out>/guidance_only/{train,test}.jsonl   Guidance-only SFT (P -> R)
    <out>/guidance_only_all_cards/...        Guidance-only SFT + all 22 cards (control)
    <out>/trajectory/{train,test}.jsonl      trajectory SFT ([P; O; E; C; R])
    <out>/card_library.json                  card text inserted at inference
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from sleepguide.stage2.trajectory import (
    all_cards,
    canonical_observation,
    export,
    observation_verdicts,
    packet_budget,
    sequences,
    threshold_flags,
)


def audit(arm: Path, library: Path) -> None:
    for module, extra in (("audit_observation", ["--source", str(arm), "--use", "target"]),
                          ("audit_sequences", ["--arm", str(arm), "--library-json", str(library)])):
        subprocess.run([sys.executable, "-m", f"sleepguide.stage2.trajectory.{module}", *extra],
                       check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/sft"))
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    steps = args.out / "steps"
    export.build(args.stage1_dir, steps / "export")
    packet_budget.build(steps / "export", steps / "packet_budget")
    sequences.build(steps / "packet_budget", steps / "sequences")
    threshold_flags.build(steps / "sequences" / "trajectory", steps / "threshold_flags")
    canonical_observation.build(steps / "threshold_flags", steps / "canonical_observation")
    observation_verdicts.build(steps / "canonical_observation", args.out / "trajectory")

    shutil.copytree(steps / "sequences" / "guidance_only", args.out / "guidance_only",
                    dirs_exist_ok=True)
    shutil.copyfile(steps / "sequences" / "card_library.json", args.out / "card_library.json")
    all_cards.build(args.out / "guidance_only", args.out / "card_library.json",
                    args.out / "guidance_only_all_cards")
    if not args.skip_audit:
        audit(args.out / "trajectory", args.out / "card_library.json")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()

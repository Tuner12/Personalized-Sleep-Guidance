#!/usr/bin/env python3
"""Stage-2 SFT from a recipe (paper Section 3.2).

    accelerate launch --num_processes 2 --mixed_precision bf16 \
        scripts/train.py --recipe recipes/qwen3.5-0.8b/trajectory_sft_seed1.yaml
"""

from sleepguide.stage2.train import main

if __name__ == "__main__":
    main()

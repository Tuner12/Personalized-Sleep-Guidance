#!/usr/bin/env python3
"""Score a generation file: Evidence F1, Value F1, ROUGE-L, BERTScore (paper Section 4.1).

    python scripts/evaluate.py --generated outputs/generations/<run>.jsonl \
        --test-file outputs/sft/trajectory/test.jsonl --stage1-dir outputs/stage1 \
        --output outputs/scores/<run>.json
"""

from sleepguide.evaluation.evaluate import main

if __name__ == "__main__":
    main()

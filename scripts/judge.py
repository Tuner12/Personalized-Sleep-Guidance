#!/usr/bin/env python3
"""LLM-as-a-Judge (Claude Opus 5, 16-item Generation Quality rubric; paper Table 1).

    python scripts/judge.py --generated outputs/generations/<run>.jsonl --out outputs/judge/<run>
"""

from sleepguide.evaluation.llm_judge import main

if __name__ == "__main__":
    main()

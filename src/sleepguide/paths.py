"""Repository-relative locations shared by the pipeline."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "prompts"
EVIDENCE_BANK = REPO / "evidence_bank"

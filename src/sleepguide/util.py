"""Small shared helpers. Kept deliberately thin so modules do not import each other."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def content_hash(value: Any) -> str:
    """Stable SHA-256 over a model or plain value, for provenance in artifacts."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@cache
def read_prompt(name: str) -> str:
    """Load a prompt by stem from the top-level prompts directory."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt {name!r} not found at {path}")
    return path.read_text(encoding="utf-8")


@cache
def read_prompt_bundle(*names: str) -> str:
    """Compose versioned prompt components in a stable, cache-friendly order."""

    return "\n\n".join(read_prompt(name).strip() for name in names)

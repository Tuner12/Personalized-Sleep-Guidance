"""Tokenize Stage-2 SFT records for Qwen3.5 with completion-only labels.

Each record is one chat turn: the user message is the profile packet P (plus instructions)
and the assistant message is the supervised completion -- `[O; E; C; R]` for trajectory SFT,
`R` alone for Guidance-only SFT. Prompt tokens are labelled -100, so the loss is the
auto-regressive cross-entropy over the completion only (paper Eq. 3). Sequences are never
truncated; a record longer than `max_length` is an error.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRAJECTORY_TASK = "trajectory_sft"
GUIDANCE_ONLY_TASK = "guidance_only_sft"


@dataclass(frozen=True)
class TokenAudit:
    examples: int
    minimum: int
    median: int
    p95: int
    maximum: int
    over_limit: int

    def model_dump(self) -> dict[str, int]:
        return asdict(self)


def load_records(path: Path, *, task: str) -> list[dict[str, Any]]:
    """Load SFT records of one task, failing on anything malformed."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            missing = {"case_id", "input", "target"} - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            if record.get("task") != task:
                raise ValueError(f"{path}:{line_number}: expected {task}, got {record.get('task')!r}")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _chat_text(tokenizer: Any, messages: list[dict[str, str]], *, generate: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generate,
        enable_thinking=False,
    )


def _token_span(
    offsets: list[tuple[int, int]], char_span: tuple[int, int]
) -> tuple[int, int] | None:
    start, end = char_span
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    if not indices:
        return None
    return min(indices), max(indices) + 1


def _assistant_bounds(full_text: str, prompt_text: str, target: str) -> tuple[int, int]:
    search_start = max(0, len(prompt_text) - 16)
    target_start = full_text.find(target, search_start)
    if target_start < 0:
        raise ValueError("chat template did not preserve the exact target")
    return target_start, target_start + len(target)


def tokenize_record(record: dict[str, Any], tokenizer: Any, *, max_length: int) -> dict[str, Any]:
    """Build one completion-only example; fail rather than truncate."""

    prompt_messages = [{"role": "user", "content": record["input"]}]
    full_messages = [*prompt_messages, {"role": "assistant", "content": record["target"]}]
    prompt_text = _chat_text(tokenizer, prompt_messages, generate=True)
    full_text = _chat_text(tokenizer, full_messages, generate=False)
    target_bounds = _assistant_bounds(full_text, prompt_text, record["target"])
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(item) for item in encoded["offset_mapping"]]
    if len(input_ids) > max_length:
        raise ValueError(
            f"case {record['case_id']!r} requires {len(input_ids)} tokens, above "
            f"max_length={max_length}; Stage 2 forbids truncation"
        )
    target_token_span = _token_span(offsets, target_bounds)
    if target_token_span is None:
        raise ValueError(f"case {record['case_id']!r}: target has no token span")
    completion_start = target_token_span[0]
    return {
        "input_ids": input_ids,
        "attention_mask": list(encoded["attention_mask"]),
        "labels": [-100] * completion_start + input_ids[completion_start:],
    }


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    position = math.ceil(fraction * len(values)) - 1
    return sorted(values)[max(0, min(position, len(values) - 1))]


def tokenize_records(
    records: Iterable[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> tuple[list[dict[str, Any]], TokenAudit]:
    examples = [tokenize_record(record, tokenizer, max_length=max_length) for record in records]
    lengths = [len(item["input_ids"]) for item in examples]
    ordered = sorted(lengths)
    audit = TokenAudit(
        examples=len(examples),
        minimum=ordered[0] if ordered else 0,
        median=_percentile(lengths, 0.5),
        p95=_percentile(lengths, 0.95),
        maximum=ordered[-1] if ordered else 0,
        over_limit=sum(length > max_length for length in lengths),
    )
    return examples, audit

"""TRL trainer and collator for Stage-2 SFT (completion-only cross-entropy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from trl import SFTTrainer


@dataclass
class SFTDataCollator:
    """Right-pad pre-tokenized examples; padded label positions are -100."""

    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        for index, item in enumerate(features):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"])
            attention_mask[index, :length] = torch.tensor(item["attention_mask"])
            labels[index, :length] = torch.tensor(item["labels"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TrajectorySFTTrainer(SFTTrainer):
    """`SFTTrainer` whose checkpoint resume understands Qwen3.5's key namespace."""

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
        """Resume with the checkpoint's multimodal key namespace remapped.

        Qwen3.5 checkpoints store text weights under ``model.language_model.*`` while the
        text-only ``Qwen3_5ForCausalLM`` expects ``model.*``. The Trainer's raw
        ``load_state_dict(strict=False)`` would match no key and silently resume from base
        weights, so the keys are remapped here and any unresolved mismatch is an error.
        """

        import safetensors.torch

        target = model if model is not None else self.model
        directory = Path(resume_from_checkpoint)
        shards = sorted(directory.glob("model-*.safetensors")) or [
            directory / "model.safetensors"
        ]
        if not all(shard.exists() for shard in shards):
            super()._load_from_checkpoint(resume_from_checkpoint, model)
            return

        state_dict: dict[str, torch.Tensor] = {}
        for shard in shards:
            state_dict.update(safetensors.torch.load_file(shard, device="cpu"))
        expected = set(target.state_dict())
        if not (set(state_dict) & expected):
            state_dict = {
                key.replace("model.language_model.", "model.", 1): value
                for key, value in state_dict.items()
            }
        if not set(state_dict) & expected:
            raise RuntimeError(
                f"{resume_from_checkpoint}: no checkpoint key matches the model even "
                "after namespace remapping; refusing to resume from base weights"
            )
        target.load_state_dict(state_dict, strict=False)
        missing = sorted(expected - set(state_dict))
        # A tied lm_head is not serialised and is restored from the embedding table.
        if missing not in ([], ["lm_head.weight"]):
            raise RuntimeError(f"{resume_from_checkpoint}: unresolved missing keys {missing[:5]}")

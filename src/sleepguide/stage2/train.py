"""Stage-2 supervised fine-tuning of Qwen3.5 from a recipe (paper Section 3.2).

    accelerate launch --num_processes 2 --mixed_precision bf16 \
        scripts/train.py --recipe recipes/qwen3.5-0.8b/trajectory_sft_seed1.yaml

The recipe fixes the effective batch size; gradient accumulation is derived from the number
of processes, so the optimisation schedule is the same on one GPU or several.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM, set_seed
from trl import SFTConfig

from sleepguide.stage2.tokenize import load_records, tokenize_records
from sleepguide.stage2.trainer import SFTDataCollator, TrajectorySFTTrainer


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = {"model", "data", "training"} - set(recipe)
    if missing:
        raise ValueError(f"{path}: missing recipe sections {sorted(missing)}")
    return recipe


def load_text_model(model_id: str, revision: str = "main") -> Qwen3_5ForCausalLM:
    """Load only the Qwen3.5 language backbone, excluding vision weights."""

    full_config = AutoConfig.from_pretrained(model_id, revision=revision)
    text_config = getattr(full_config, "text_config", None)
    if text_config is None and full_config.model_type == "qwen3_5_text":
        # A checkpoint written by this code has already been reduced to the text config.
        text_config = full_config
    if text_config is None or text_config.model_type != "qwen3_5_text":
        raise ValueError(f"{model_id!r} is not a dense Qwen3.5 checkpoint")
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_id,
        config=text_config,
        revision=revision,
        dtype=torch.bfloat16,
        use_safetensors=True,
    )
    model.config.use_cache = False
    return model


def gradient_accumulation(effective_batch: int, per_device: int) -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    steps, remainder = divmod(effective_batch, per_device * world)
    if remainder or steps < 1:
        raise ValueError(
            f"effective_batch_size={effective_batch} is not a multiple of "
            f"per_device_train_batch_size={per_device} x {world} processes"
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--token-audit-only", action="store_true")
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    model_id, revision = recipe["model"], recipe.get("revision") or "main"
    data, train = recipe["data"], recipe["training"]
    set_seed(train["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("span alignment requires a fast tokenizer with offset mappings")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(Path(data["train_file"]), task=data["task"])
    examples, audit = tokenize_records(records, tokenizer, max_length=train["max_length"])
    train_dataset = Dataset.from_list(examples)
    output_dir = Path(train["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "token_audit.json").write_text(
        json.dumps(audit.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit.model_dump(), indent=2, sort_keys=True))
    if args.token_audit_only:
        return

    per_device = train["per_device_train_batch_size"]
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train["num_train_epochs"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=gradient_accumulation(train["effective_batch_size"], per_device),
        learning_rate=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
        warmup_ratio=float(train["warmup_ratio"]),
        lr_scheduler_type=train["lr_scheduler_type"],
        max_grad_norm=float(train["max_grad_norm"]),
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train["logging_steps"],
        eval_strategy="no",
        save_strategy="steps",
        save_steps=train["save_steps"],
        save_total_limit=train["save_total_limit"],
        # Off: the Trainer's best-model reload does not remap Qwen3.5's key namespace.
        load_best_model_at_end=False,
        report_to=train.get("report_to", "none"),
        seed=train["seed"],
        data_seed=train["seed"],
        remove_unused_columns=False,
        prediction_loss_only=True,
        packing=False,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        deepspeed=train.get("deepspeed"),
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused",
        loss_type="chunked_nll",
    )
    trainer = TrajectorySFTTrainer(
        model=load_text_model(model_id, revision),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

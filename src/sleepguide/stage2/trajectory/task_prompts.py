"""Versioned Stage-2 task instructions, read by the export step."""

from __future__ import annotations

from sleepguide.util import read_prompt, read_prompt_bundle

STAGE2_PROMPT_VERSION = "stage2-prompts-1.0"

_TASK_PROMPT_FILES = {
    "predict_case_observation": ("stage2/observation",),
    "select_case_evidence_from_observation": ("stage2/evidence_selector",),
    "write_four_section_recommendation_from_evidence": (
        "shared/guidance_contract",
        "stage2/evidence_writer",
        "shared/reader_language_contract",
    ),
    "write_four_section_recommendation_from_profile": (
        "shared/guidance_contract",
        "stage2/guidance_only",
        "shared/reader_language_contract",
    ),
}


def stage2_task_prompt(task: str) -> str:
    """Return the canonical instruction used for both SFT export and inference."""

    names = _TASK_PROMPT_FILES.get(task)
    if names is None:
        raise KeyError(f"unknown Stage 2 task: {task}")
    if len(names) == 1:
        return read_prompt(names[0]).strip()
    return read_prompt_bundle(*names)


def stage2_prompt_catalog() -> dict[str, dict[str, str]]:
    return {
        task: {
            "prompt_version": STAGE2_PROMPT_VERSION,
            "instruction": stage2_task_prompt(task),
        }
        for task in _TASK_PROMPT_FILES
    }

import json

import pytest

from sleepguide.paths import EVIDENCE_BANK
from sleepguide.stage2 import best_of_n
from sleepguide.stage2.tokenize import TRAJECTORY_TASK, load_records


def test_best_of_n_prefers_axis_agreement_when_verdicts_tie(monkeypatch):
    parsed = {"a": ({("total_sleep_min", "recent")}, set()),
              "b": ({("waso_min", "recent")}, set())}
    monkeypatch.setattr(best_of_n, "axes_and_verdicts", lambda text: parsed[text])
    assert best_of_n.select(["a", "b"], {("waso_min", "recent")}) == 1


def test_best_of_n_prefers_verdict_consensus_when_axes_tie(monkeypatch):
    axes = {("total_sleep_min", "recent")}
    parsed = {"x": (axes, {"short"}), "y": (axes, {"short"}), "z": (axes, {"long"})}
    monkeypatch.setattr(best_of_n, "axes_and_verdicts", lambda text: parsed[text])
    assert best_of_n.select(["z", "x", "y"], axes) in (1, 2)


def test_parse_identifiers_drops_unknown_and_duplicates():
    pytest.importorskip("torch")
    from sleepguide.stage2.generate import parse_identifiers

    cards = {"allow-more-time-for-sleep": "...", "make-sleep-timing-more-consistent": "..."}
    head = ("<OBSERVATION>\no\n</OBSERVATION>\n<EVIDENCE>\nallow-more-time-for-sleep\n"
            "invented-card\nallow-more-time-for-sleep\nmake-sleep-timing-more-consistent\n")
    names, invalid = parse_identifiers(head, cards)
    assert names == ["allow-more-time-for-sleep", "make-sleep-timing-more-consistent"]
    assert invalid == 1


def test_generation_bank_has_the_22_runtime_cards():
    bank = json.loads((EVIDENCE_BANK / "generation_bank.json").read_text())
    assert len(bank) == 22 and len({card["evidence_key"] for card in bank}) == 22


def test_load_records_rejects_the_wrong_task(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"case_id": "c", "input": "P", "target": "R",
                                "task": "guidance_only_sft"}) + "\n")
    with pytest.raises(ValueError):
        load_records(path, task=TRAJECTORY_TASK)


def test_baseline_prompt_carries_all_cards_before_the_closing_instruction():
    from sleepguide.baselines.llm_prompting import build_prompt

    cards = {"card-a": "[card-a] text A", "card-b": "[card-b] text B"}
    record = "CONTRACT\n\nInput payload (JSON):\n\n{}\n\nReturn only the four tagged segments."
    plain = build_prompt(record, "gpt", cards)
    health = build_prompt(record, "gpt_health", cards)
    assert plain.index("[card-a] text A") < plain.index("[card-b] text B") < plain.index("Return only")
    assert plain.endswith("</SECTIONS>") and "Produce EXACTLY these three tagged segments" in plain
    assert health.endswith(plain) and health.startswith("You are a careful consumer-health")


def test_judge_prompt_lists_every_case_and_all_16_items():
    from sleepguide.evaluation.llm_judge import KEYS, build_prompt

    prompt = build_prompt(["=== CASE 1 ===", "=== CASE 2 ==="])
    assert "grading 2 candidate" in prompt and "JSON array of 2 objects" in prompt
    assert all(f'"{k}":_' in prompt for k in KEYS) and len(KEYS) == 16


def test_profile_agent_request_and_validation(tmp_path):
    from sleepguide.stage1.profile_agent import build_request, validate

    request = build_request(tmp_path / "records", tmp_path / "profiles", "field notes")
    assert "# Profile Agent" in request and "## JSON Schema of each output file" in request
    assert '"metric_id": "total_sleep_min"' in request
    (tmp_path / "case_x.json").write_text('{"profile": {"case_id": "case_x"}}')
    valid, errors = validate(tmp_path)
    assert valid == 0 and "case_x" in errors

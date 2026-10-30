from sleepguide.core.recommendation import _anchor_value_is_visible
from sleepguide.stage1 import CaseObservation
from sleepguide.stage2.trajectory.task_prompts import STAGE2_PROMPT_VERSION, stage2_task_prompt

def test_anchor_visibility_accepts_half_minute_medians_in_reader_display() -> None:
    assert _anchor_value_is_visible("typically 334.5 min", 334.5, "min")
    assert _anchor_value_is_visible("around 1:02 a.m.", 62.5, "clock_minute")
    assert _anchor_value_is_visible("around 1:03 a.m.", 62.5, "clock_minute")
    assert _anchor_value_is_visible("around 1:02:30 a.m.", 62.5, "clock_minute")


def test_stage2_recommendation_routes_share_the_same_section_contract() -> None:
    evidence_prompt = stage2_task_prompt(
        "write_four_section_recommendation_from_evidence"
    )
    pure_prompt = stage2_task_prompt("write_four_section_recommendation_from_profile")
    shared_markers = (
        "## Section 1: Summary",
        "## Section 2: Pattern Analysis",
        "## Section 3: Personalized Recommendation",
        "## Section 4: Follow-up",
    )
    for marker in shared_markers:
        assert marker in evidence_prompt
        assert marker in pure_prompt
    assert "Evidence E" in evidence_prompt
    assert "directly to Profile P" in pure_prompt
    assert STAGE2_PROMPT_VERSION.startswith("stage2-prompts-1.0")
    assert "# Reader-Language Contract v1" in evidence_prompt
    assert "# Reader-Language Contract v1" in pure_prompt


def test_compact_observation_contract_does_not_duplicate_longitudinal_views() -> None:
    fields = set(CaseObservation.model_fields)
    assert fields == {
        "case_id",
        "primary_observation",
        "supporting_observations",
        "context_observation",
        "relevant_metric_ids",
        "anchors",
    }
    observation_prompt = stage2_task_prompt("predict_case_observation")
    assert "Do not reproduce all longitudinal views" in observation_prompt
    assert "interpretive question" in observation_prompt
    assert "longitudinal_summary" not in observation_prompt
    assert "decision_focus" not in observation_prompt

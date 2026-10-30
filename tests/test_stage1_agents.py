from sleepguide.llm import _decode_json
from sleepguide.stage1 import RecommendationDraft, SectionGrounding
from sleepguide.stage1.construct import _editable_sections, _successful_polish_for_writer


def test_decode_json_uses_last_complete_object_when_model_repeats_output() -> None:
    assert _decode_json('{"case_id":"old"}\n{"case_id":"corrected"}') == {
        "case_id": "corrected"
    }


def test_writer_feedback_recovers_section_names_without_strict_prefixes() -> None:
    assert _editable_sections(
        [
            {"reason": "Section 2 repeats an unobserved conditional factor"},
            {"reason": "section_1 uses average language without a mean anchor"},
        ]
    ) == {"section_1", "section_2"}


def _draft(summary: str) -> RecommendationDraft:
    return RecommendationDraft(
        case_id="case_0001",
        section_1=summary,
        section_2="Analysis.",
        section_3="Recommendation.",
        section_4="Follow-up.",
        section_1_grounding=SectionGrounding(),
        section_2_grounding=SectionGrounding(),
        section_3_grounding=SectionGrounding(),
        section_4_grounding=SectionGrounding(),
    )


def test_successful_language_polish_is_restored_only_for_its_exact_writer_draft() -> None:
    source = _draft("Source summary.")
    polished = _draft("Polished summary.")
    attempts = [
        {
            "revision": 2,
            "stage": "language_polish",
            "issues": [],
            "source_recommendation_draft": source.model_dump(mode="json"),
            "polished_recommendation_draft": polished.model_dump(mode="json"),
        }
    ]

    assert _successful_polish_for_writer(attempts, 2, source) == polished
    assert _successful_polish_for_writer(attempts, 1, source) is None
    assert _successful_polish_for_writer(attempts, 2, _draft("Different source.")) is None

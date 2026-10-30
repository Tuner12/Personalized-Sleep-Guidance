"""The Stage-1 agents and their deterministic checks (paper Section 3.1, Figure 2).

    Evidence Agent     plan_case              observation O and evidence identifiers E
    Composer Agent     write_recommendation   four-section guidance R with numeric anchors
                       polish_recommendation  reader-language revision of a draft
    deterministic      validate_observation, validate_selection, validate_recommendation
    Reflection Agent   review_blind_reverse   recovers the required evidence from R and P alone

The verification loop that sequences these calls is in `construct.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sleepguide.core.contracts import LongitudinalInsight, SleepProfile
from sleepguide.core.recommendation import (
    AnchorOrigin,
    AnchorStatistic,
    CaseMetricAnchor,
    ReaderRecommendation,
    _anchor_value_is_visible,
    resolve_insight_anchor,
    resolve_profile_anchor,
)
from sleepguide.llm import LLMClient
from sleepguide.core.packet import generation_case_signals, generation_model_input
from sleepguide.stage1.contracts import (
    BlindReverseReview,
    CaseObservation,
    EvidenceCardLibrary,
    EvidenceKind,
    EvidenceReferenceAnchor,
    EvidenceSelection,
    RecommendationDraft,
    ReverseComparison,
    ReverseIssue,
    ReverseIssueType,
    ReverseResult,
    SectionGrounding,
    Stage1Plan,
)
from sleepguide.stage1.evidence_cards import build_case_evidence, writer_card
from sleepguide.stage1.numeric import _insight_numeric_spans
from sleepguide.stage1.candidate_evidence import candidate_evidence_pools
from sleepguide.util import read_prompt, read_prompt_bundle

_INTERNAL_READER_LANGUAGE = {
    "rolling": r"\brolling\b",
    "block": r"\bblocks?\b",
    "centered": r"\bcenter(?:ed)?\b",
    "baseline": r"\bbaseline\b",
    "multi-night": r"\bmulti[- ]night\b",
    "provisional": r"\bprovisional\b",
    "trajectory": r"\btrajectory\b",
    "research narration": (
        r"\b(?:research|evidence|stud(?:y|ies))\s+(?:shows?|suggests?|supports?)\b"
    ),
    "collection technology": (
        r"\b(?:tracker|wearable|fitbit|device|sensor|wrist[- ]based|"
        r"consumer sleep records?)\b"
    ),
    "statistical jargon": r"\b(?:spearman|rho|MAD|IQR|correlation coefficient)\b",
    "research protocol language": r"\b(?:intervention|protocol|experiment|reassess)\b",
    "ambiguous internal period language": r"\b(?:sleep windows?|along the way)\b",
    "technical sleep-opportunity language": r"\bsleep opportunity\b",
    "awkward time-in-bed wording": r"\btime available in bed\b",
    "awkward follow-up wording": r"\blook for whether\b",
    "ambiguous awake-gap wording": r"\bawake gap\b",
    "automated-estimate wording": r"\bautomated estimate\b",
    "automated-classification wording": r"\bautomated\b",
    "ISO date": r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b",
    "ambiguous continuity change": r"\b(?:your )?(?:recent )?sleep was less continuous\b",
    "abstract continuity label": r"\b(?:reduced|lower|limited) (?:sleep )?continuity\b",
    "retrospective follow-up grammar": r"\bit helped if\b",
    "indirect clinical next-step wording": (
        r"\b(?:would|could|may) be (?:a )?reason to "
        r"(?:speak|talk|discuss|consult|contact|seek)\b|"
        r"\b(?:supports?|would support) "
        r"(?:speaking|talking|discussing|consulting|contacting|seeking)\b"
    ),
    "formulaic therefore connector": r"\btherefore\b",
    "repeated while linkage": r"\bwhile\b[^.!?]*\bwhile\b",
    "compressed even-as linkage": r"\beven as\b",
    "abstract awake-time trend wording": (
        r"\btime awake after falling asleep\s+"
        r"(?:increased|decreased|rose|fell|was reduced)\b"
    ),
    "awkward lower-amount comparison": (
        r"\b(?:usual|general) lower amount (?:associated with|for)\b"
    ),
    "unnecessary one-night action reversal": (
        r"\bdo not change\b[^.]{0,80}\b(?:one|single|latest|unusual) (?:night|date)\b|"
        r"\binterpret (?:an? )?(?:isolated|different|unusual) night in context\b"
    ),
    "indirect monitoring judgment": r"\b(?:remains?|is|are) worth watching\b",
    "invented routine linkage": (
        r"\b(?:parts? of )?(?:your|the) (?:current|usual) routine that "
        r"(?:coincided|accompanied|went along)\b"
    ),
    "audit-style tradeoff judgment": (
        r"\bdoes not outweigh\b[^.]{0,100}\b(?:decid|decision|maintain)\w*\b"
    ),
    "abstract continuity inference": (
        r"\b(?:indicate[sd]?|suggest[sd]?) (?:that (?:your|the) sleep (?:was|is) )?"
        r"less continuous\b"
    ),
    "compressed interrupted-sleep wording": (
        r"\b(?:fits?|fit|matches?)\s+(?:more|less)\s+interrupted\s+sleep\b"
    ),
    "indirect outcome verdict": (
        r"\b(?:a|the) (?:repeated|consistent) "
        r"(?:increase|decrease|change|improvement) would (?:show|indicate)\b"
    ),
    "audit-style departure wording": r"\b(?:a |the )?clear departure from\b",
    "dry worth-addressing wording": r"\bworth addressing\b",
    "dry contributor wording": r"\b(?:was|is) an important contributor to\b",
    "awkward continuity-reference wording": (
        r"\breference(?:\s+of\s+[^.]{0,40})?\s+for\s+less continuous sleep\b"
    ),
    "reader-facing reference jargon": (
        r"\b(?:population reference|above the (?:general |adult |older-adult )?"
        r"(?:sleep-quality )?reference|below the (?:general |adult |older-adult )?"
        r"(?:sleep-quality )?reference|reference of (?:at least )?\d+|"
        r"(?:above|below) the (?:general |adult |older-adult )?"
        r"(?:sleep-quality )?(?:benchmark|threshold)(?: of)?)\b"
    ),
    "reader-facing reference term": r"\breference\b",
    "reader-facing numeric age band": (
        r"\b(?:aged\s+)?(?:18\s+(?:to|-)\s+64|65\s+and older)\b"
    ),
    "vague daytime-problem wording": r"\bdaytime (?:problems?|effects?)\b",
    "technical midpoint wording": r"\b(?:sleep[- ]?)?midpoint\b",
    "unnecessary record-count framing": r"\bacross\s+\d+\s+nights?\b",
    "abstract care wording": r"\b(?:outside review|clinical context)\b",
    "laboratory comparison wording": r"\b(?:laboratory measurement|polysomnography)\b",
    "inverted general-sign comparison": (
        r"\b(?:below|above|less than|more than)\s+[^,.]{0,30},\s+"
        r"which is one general sign\b"
    ),
    "estimate-versus-diagnosis wording": r"\bestimate rather than a diagnosis\b",
    "qualified WASO wording": (
        r"\bestimat(?:e|ed)\b[^.]{0,40}\b(?:awake|wakefulness)\b|"
        r"\b(?:awake time|wakefulness)\b[^.]{0,20}\bis an estimate\b"
    ),
    "unsupported sleep-time withdrawal": (
        r"\bstop (?:adding|allowing|leaving|making) (?:more )?time "
        r"(?:in bed|for sleep)\b|\bstop moving (?:sleep|bedtime) earlier\b"
    ),
    "abstract strategy verdict": (
        r"\bstop treating\b[^.]{0,80}\b(?:sleep(?:-specific)? strategy|sleep-related action)\b|"
        r"\bstop changing\b[^.]{0,60}\bfor this purpose\b|"
        r"\bstop using\b[^.]{0,60}\bfor the purpose of\b"
    ),
    "raw metric identifier": (
        r"\b(?:waso_min|minute_to_fall_asleep|minute_after_wakeup|count_wake|"
        r"wake_episode_over_5_min_count|wake_episode_count|longest_wake_episode_min|"
        r"sleep_onset_clock_min|final_wake_clock_min|sleep_midpoint_clock_min)\b"
    ),
    "minute-form seven-hour comparison": r"\b420\s+minutes?\b",
    "awkward seven-hour comparison": r"\b(?:far|well)\s+below\s+7\s+hours\s+or\s+more\b",
    "sleep-opportunity phrasing": r"\b(?:little|limited)\s+opportunity\s+for\s+(?:longer|more)\s+sleep\b",
    "unspecified better-or-worse language": (
        r"\b(?:the latest night|sleep|the pattern)\s+(?:was|became)\s+"
        r"(?:better|worse)\b"
    ),
    "vague whole-record variation": (
        r"\b(?:across|over) (?:the )?(?:available|recorded) nights,? "
        r"your (?:sleep duration|sleep timing|sleep efficiency) varied\b"
    ),
    "unsupported absence of concern": (
        r"\bno (?:repeated |remaining |clear )?(?:concern|issue|problem) "
        r"(?:is|was|has been) (?:identified|present|found)\b"
    ),
    "awkward unequal-duration wording": r"\bsleep was not equally long\b",
    "rejected-action narration": (
        r"\b(?:the (?:record|data)|your (?:record|data))\s+does not\s+"
        r"(?:support|justify)\b|"
        r"\bthere (?:is|was) no need to\s+(?:increase|extend|change|move)\b|"
        r"\bsolely to (?:add|reach|meet)\b|"
        r"\brather than (?:increasing|extending|moving|changing)\b|"
        r"\bmore useful than (?:changing|increasing|extending|moving)\b"
    ),
    "template report framing": (
        r"\b(?:the main pattern is|the strongest counterpoint is|"
        r"the record supports|the data suggest)\b"
    ),
    "unsupported causal attribution": (
        r"\b(?:sleep|the pattern|this pattern)\s+(?:is|was)\s+"
        r"(?:caused|driven|explained)\s+by\b"
    ),
    "research-style significance wording": r"\bclinical significance\b",
    "abstract clinical-context judgment": (
        r"\b(?:problems?|patterns?)\s+(?:deserve|warrant)\s+"
        r"(?:clinical context|attention|review)\b"
    ),
    "latest-night inference wording": (
        r"\b(?:one(?: short)? night|the latest night|that night)\s+"
        r"(?:did not |does not |failed to )?(?:by itself\s+)?"
        r"(?:resolve|confirm|establish|require)\b"
    ),
    "dry decision wording": r"\bproportionate\b",
    "audit-style variation wording": (
        r"\b(?:(?:dated|single) variation|isolated departure)\b"
    ),
    "reference-as-cause wording": (
        r"\b(?:less continuous|reduced continuity)\s+because\s+"
        r"[^.]{0,100}\breference\b"
    ),
}

_LANGUAGE_REPAIR_HINTS = {
    "technical sleep-opportunity language": (
        "replace it with `time available for sleep` or the exact `time in bed` measure"
    ),
    "awkward time-in-bed wording": (
        "say `even though you spent enough time in bed` or name the exact time in bed"
    ),
    "ISO date": "write the date as `Month D`, for example `November 16`",
    "rejected-action narration": (
        "remove the rejected alternative and state the observed sleep finding directly"
    ),
    "automated-estimate wording": "write `time awake after falling asleep`",
    "automated-classification wording": "write `time awake after falling asleep`",
    "abstract continuity label": (
        "name the observable measure, such as more time awake after falling asleep, or say "
        "plainly that sleep was more interrupted"
    ),
    "latest-night inference wording": (
        "describe whether the latest night fit or differed from nearby nights without a "
        "review judgment"
    ),
    "dry decision wording": (
        "connect the recommendation directly to the recent sleep pattern in ordinary language"
    ),
    "audit-style variation wording": (
        "name what happened on the night in ordinary language and include its date when it matters"
    ),
    "vague whole-record variation": (
        "state the usual whole-record level or the direction of the material change instead "
        "of saying only that a metric varied"
    ),
    "unsupported absence of concern": (
        "state which recorded dimension was adequate, stable, or improving; the absence of "
        "recorded deterioration does not establish that the person has no concern"
    ),
    "awkward unequal-duration wording": (
        "state directly that sleep duration was shorter than time in bed or give the two "
        "values when their difference matters"
    ),
    "reference-as-cause wording": (
        "state the person's value, then give the comparison as one general sign of sleep "
        "quality; do not make the comparison a cause"
    ),
    "reader-facing reference jargon": (
        "state the person's value first, then explain the comparison plainly, such as "
        "`For older adults, 30 minutes or less is commonly used as one general sign of "
        "good sleep quality`"
    ),
    "reader-facing reference term": (
        "state the comparison as one general sign of good sleep quality without naming a "
        "reference"
    ),
    "reader-facing numeric age band": (
        "write `adults` or `older adults` without copying a numeric age band"
    ),
    "vague daytime-problem wording": (
        "name the missing context directly, such as unusual sleepiness during the day or "
        "trouble concentrating"
    ),
    "technical midpoint wording": (
        "write `the middle of the sleep period` or describe the whole sleep period shift"
    ),
    "unnecessary record-count framing": (
        "start with the person's usual sleep pattern; omit the total number of recorded nights"
    ),
    "abstract care wording": (
        "say directly when it would be useful to discuss the pattern with a clinician"
    ),
    "laboratory comparison wording": "write `time awake after falling asleep`",
    "qualified WASO wording": (
        "write `time awake after falling asleep` without `estimated` or another measurement qualifier"
    ),
    "minute-form seven-hour comparison": (
        "write `7 hours` rather than `420 minutes` when describing the seven-hour sleep "
        "duration comparison"
    ),
    "awkward seven-hour comparison": (
        "write `well below 7 hours` or `below 7 hours`; do not combine `below` with "
        "`7 hours or more`"
    ),
    "sleep-opportunity phrasing": (
        "write directly that less time in bed limited sleep duration, without saying "
        "there was little or limited opportunity for longer sleep"
    ),
    "unsupported sleep-time withdrawal": (
        "state whether added sleep time showed a clear effect; do not instruct the reader to "
        "withdraw reasonable time for sleep merely because duration did not improve"
    ),
    "abstract strategy verdict": (
        "state that the tested factor may not be an important contributor to the current "
        "sleep pattern; add professional consultation only when selected clinical-boundary "
        "Evidence supports it"
    ),
    "inverted general-sign comparison": (
        "make the comparison itself the subject, such as `85% or more is one general sign`"
    ),
    "estimate-versus-diagnosis wording": (
        "separate measurement uncertainty from the statement that the pattern alone cannot "
        "diagnose a sleep disorder"
    ),
    "retrospective follow-up grammar": (
        "write prospectively, such as `You will know this is helping if...`"
    ),
    "indirect clinical next-step wording": (
        "state the condition and action directly, such as `If the difficulty continues or "
        "affects daily life, speak with a doctor`"
    ),
    "formulaic therefore connector": (
        "remove `therefore`; use two direct sentences or name the relationship explicitly"
    ),
    "repeated while linkage": (
        "use `while` for at most one necessary contrast in a sentence; otherwise use `and`, "
        "`but`, or separate sentences"
    ),
    "compressed even-as linkage": (
        "preserve the contrast but use two direct sentences or a simple `but`"
    ),
    "abstract awake-time trend wording": (
        "make the person the subject: write `you spent less time awake after falling "
        "asleep` or `you spent more time awake after falling asleep`"
    ),
    "awkward lower-amount comparison": (
        "state the population context directly, such as `Older adults usually need at "
        "least 7 hours of sleep, although individual needs vary`"
    ),
    "unnecessary one-night action reversal": (
        "remove the rejected one-night alternative; state the repeated pattern and the "
        "action to keep directly"
    ),
    "indirect monitoring judgment": (
        "state which sleep dimension remained unresolved; leave the action or evaluation "
        "to Personalized Recommendation or Follow-up"
    ),
    "invented routine linkage": (
        "name the supported action directly, such as leaving a similar amount of time for "
        "sleep; do not invent a routine that was not recorded"
    ),
    "audit-style tradeoff judgment": (
        "state directly which dimensions improved and which did not; do not narrate how the "
        "pipeline weighed them"
    ),
    "abstract continuity inference": (
        "say directly that the person spent more time awake after falling asleep or that "
        "sleep was more interrupted"
    ),
    "compressed interrupted-sleep wording": (
        "write `suggests that sleep was more interrupted` or name the observed awake-time "
        "measure directly"
    ),
    "multiple contrast links in one sentence": (
        "keep the necessary contrast and split the remaining facts into direct sentences"
    ),
    "overloaded multi-metric sentence": (
        "split the sentence so each sentence carries one comparison or one clear contrast"
    ),
    "indirect outcome verdict": (
        "state the feedback directly, such as `If sleep lasts longer, the added time is helping`"
    ),
    "audit-style departure wording": (
        "say directly that the night was longer, shorter, earlier, or later than nearby nights"
    ),
    "dry worth-addressing wording": (
        "state the sleep concern directly and leave the action to Personalized Recommendation"
    ),
    "dry contributor wording": (
        "describe the observed relationship directly without calling a metric a contributor"
    ),
}

_INTERMEDIATE_INTERNAL_LANGUAGE = {
    "Evidence process language": (
        r"\b(?:the evidence|this evidence|this card|selected card|retrieval|routing|"
        r"grounding|matched outcomes?)\b"
    ),
    "conditional taxonomy language": r"\bconditional (?:guidance|evidence|card)\b",
    "pipeline action language": (
        r"\b(?:direct measured-data step|measured-data step|self-directed step)\b"
    ),
}

_DETERMINISTIC_APPLICABILITY_KEYS = frozenset(
    {
        "adults-18-to-64-usually-need-at-least-7-hours",
        "adults-65-plus-usually-need-at-least-7-hours",
        "sleep-efficiency-below-85-percent-may-reflect-poor-continuity",
        "waso-quality-reference-for-adults-18-to-64",
        "waso-quality-reference-for-adults-65-plus",
        "sleep-latency-quality-reference-for-adults",
        "long-wake-episode-reference-for-adults-18-to-64",
        "long-wake-episode-reference-for-adults-65-plus",
        "allow-more-time-for-sleep",
        "make-sleep-timing-more-consistent",
        "preserve-recovery-sleep-when-weekday-sleep-is-short",
        "maintain-a-stable-adequate-sleep-pattern",
        "caffeine-close-to-sleep-may-disrupt-sleep",
        "late-light-or-screen-use-may-delay-sleep",
        "a-quiet-wind-down-may-support-sleep",
        "regular-daytime-movement-may-support-sleep-efficiency",
        "sleep-stage-estimates-are-not-action-targets",
        "sleep-timing-alone-does-not-confirm-circadian-phase",
        "unwanted-sleep-timing-may-need-clinical-review",
        "persistent-short-sleep-needs-clinical-context",
        "persistent-sleep-difficulty-needs-clinical-context",
    }
)

_TIMING_EVIDENCE_KEYS = frozenset(
    {
        "make-sleep-timing-more-consistent",
        "preserve-recovery-sleep-when-weekday-sleep-is-short",
        "late-light-or-screen-use-may-delay-sleep",
        "sleep-timing-alone-does-not-confirm-circadian-phase",
        "unwanted-sleep-timing-may-need-clinical-review",
    }
)

_ACTION_TITLE = re.compile(r"(?m)^\*\*([^*\n]+)\*\*\s*$")
_EXPLICIT_CONDITION_LANGUAGE = re.compile(
    r"\b(?:if|when|whenever|unless|provided(?: that)?|as long as|only when)\b",
    re.IGNORECASE,
)
_CONTRAST_LINK = re.compile(
    r"\b(?:while|although|though|whereas|however|yet|but|even though|even as)\b",
    re.IGNORECASE,
)
_CHANGE_OR_COMPARISON = re.compile(
    r"\b(?:increase[ds]?|decrease[ds]?|rose|fell|shorter|longer|later|earlier|"
    r"improve[ds]?|worsen(?:ed|s)?|change[ds]?|move[ds]?|remain(?:ed|s)?|stay(?:ed|s)?|"
    r"versus|compared|from)\b",
    re.IGNORECASE,
)
_READER_METRIC_FAMILIES = (
    re.compile(
        r"\b(?:sleep duration|hours? (?:of )?sleep|minutes? (?:of )?sleep|slept)\b",
        re.I,
    ),
    re.compile(r"\btime in bed\b", re.I),
    re.compile(r"\btime awake after falling asleep\b", re.I),
    re.compile(
        r"\b(?:sleep efficiency|share of time (?:in bed )?spent asleep|"
        r"share of time asleep)\b",
        re.I,
    ),
    re.compile(r"\b(?:sleep onset|sleep start|sleep began|began sleep)\b", re.I),
    re.compile(r"\b(?:final wake|wake time|sleep ended|ended sleep)\b", re.I),
    re.compile(
        r"\b(?:middle of (?:your|the) sleep period|whole sleep period)\b", re.I
    ),
    re.compile(r"\bwake episodes?\b", re.I),
    re.compile(r"\b(?:activity|active days?|daytime movement)\b", re.I),
)

_NATURAL_WHOLE_SLEEP_PERIOD = re.compile(
    r"\b(?:the middle of (?:your|the) sleep period|your whole sleep period|"
    r"the whole sleep period)\b",
    re.I,
)

_UNOBSERVED_CONDITION_TERMS = {
    "caffeine-close-to-sleep-may-disrupt-sleep": re.compile(r"\bcaffeine\b", re.I),
    "late-light-or-screen-use-may-delay-sleep": re.compile(
        r"\b(?:bright light|screen(?: use)?)\b", re.I
    ),
    "a-quiet-wind-down-may-support-sleep": re.compile(
        r"\b(?:busy|stimulating|wind[- ]down)\b", re.I
    ),
    "persistent-sleep-difficulty-needs-clinical-context": re.compile(
        r"\b(?:ongoing sleep difficulty|unusual daytime sleepiness|trouble concentrating)\b",
        re.I,
    ),
    "persistent-short-sleep-needs-clinical-context": re.compile(
        r"\b(?:short sleep continues|unusual daytime sleepiness|wake unrefreshed|"
        r"trouble concentrating)\b",
        re.I,
    ),
}

_MAINTENANCE_EVIDENCE_KEY = "maintain-a-stable-adequate-sleep-pattern"
_TIMING_CLINICAL_EVIDENCE_KEY = "unwanted-sleep-timing-may-need-clinical-review"


def _language_issue(scope: str, label: str) -> str:
    issue = f"{scope} contains {label}"
    hint = _LANGUAGE_REPAIR_HINTS.get(label)
    return f"{issue}; {hint}" if hint else issue


def _reader_sentence_style_issues(text: str) -> set[str]:
    """Detect structural reading burden that a word blacklist cannot capture."""

    issues: set[str] = set()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    for sentence in sentences:
        if len(_CONTRAST_LINK.findall(sentence)) > 1:
            issues.add("multiple contrast links in one sentence")
        metric_count = sum(
            bool(pattern.search(sentence)) for pattern in _READER_METRIC_FAMILIES
        )
        direction_count = len(_CHANGE_OR_COMPARISON.findall(sentence))
        if (
            metric_count >= 4 and _CONTRAST_LINK.search(sentence)
        ) or (metric_count >= 3 and direction_count >= 4):
            issues.add("overloaded multi-metric sentence")
    return issues

_INSIGHT_ANCHOR_STATISTICS = {
    AnchorStatistic.RECENT_7_CENTER,
    AnchorStatistic.PREVIOUS_7_CENTER,
    AnchorStatistic.EARLIER_HISTORY_CENTER,
    AnchorStatistic.RECENT_MINUS_PREVIOUS_7,
    AnchorStatistic.RECENT_MINUS_EARLIER_HISTORY,
}

_AGGREGATE_ANCHOR_STATISTICS = {
    AnchorStatistic.N,
    AnchorStatistic.MEAN,
    AnchorStatistic.MEDIAN,
    AnchorStatistic.SD,
    AnchorStatistic.MAD,
    AnchorStatistic.IQR,
    AnchorStatistic.P10,
    AnchorStatistic.P90,
    AnchorStatistic.MINIMUM,
    AnchorStatistic.MAXIMUM,
    AnchorStatistic.FIRST_VALUE,
    AnchorStatistic.LAST_VALUE,
}
_LONGITUDINAL_ANCHOR_STATISTICS = {
    AnchorStatistic.SPAN_DAYS,
    AnchorStatistic.SAMPLING_DENSITY_PCT,
    AnchorStatistic.EARLIER_WINDOW_CENTER,
    AnchorStatistic.EARLIER_WINDOW_MAD,
    AnchorStatistic.EARLIER_WINDOW_IQR,
    AnchorStatistic.EARLIER_WINDOW_N,
    AnchorStatistic.EARLIER_WINDOW_CALENDAR_SPAN_DAYS,
    AnchorStatistic.RECENT_WINDOW_CENTER,
    AnchorStatistic.RECENT_WINDOW_MAD,
    AnchorStatistic.RECENT_WINDOW_IQR,
    AnchorStatistic.RECENT_WINDOW_N,
    AnchorStatistic.RECENT_WINDOW_CALENDAR_SPAN_DAYS,
    AnchorStatistic.RECENT_MINUS_EARLIER,
    AnchorStatistic.RECENT_MINUS_EARLIER_MAD,
    AnchorStatistic.SLOPE_PER_7_DAYS,
}
_WEEKPART_ANCHOR_STATISTICS = {
    AnchorStatistic.WEEKDAY_N,
    AnchorStatistic.WEEKEND_N,
    AnchorStatistic.WEEKDAY_CENTER,
    AnchorStatistic.WEEKEND_CENTER,
    AnchorStatistic.WEEKEND_MINUS_WEEKDAY,
}
_ASSOCIATION_ANCHOR_STATISTICS = {
    AnchorStatistic.PAIRED_NIGHTS,
    AnchorStatistic.SPEARMAN_RHO,
    AnchorStatistic.HIGH_MINUS_LOW_SLEEP,
}


def _canonicalize_anchor_origin(anchor: CaseMetricAnchor) -> CaseMetricAnchor:
    """Repair an unambiguous metadata origin without changing a value or claim."""

    if anchor.statistic in _INSIGHT_ANCHOR_STATISTICS:
        return anchor.model_copy(update={"origin": AnchorOrigin.INSIGHT})
    if anchor.statistic in _AGGREGATE_ANCHOR_STATISTICS:
        return anchor.model_copy(update={"origin": AnchorOrigin.AGGREGATE})
    if anchor.statistic in _LONGITUDINAL_ANCHOR_STATISTICS:
        return anchor.model_copy(update={"origin": AnchorOrigin.LONGITUDINAL})
    if anchor.statistic in _WEEKPART_ANCHOR_STATISTICS:
        return anchor.model_copy(update={"origin": AnchorOrigin.WEEKPART})
    if anchor.statistic in _ASSOCIATION_ANCHOR_STATISTICS:
        return anchor.model_copy(update={"origin": AnchorOrigin.ASSOCIATION})
    if anchor.statistic is AnchorStatistic.VALUE and anchor.observation_day_index:
        return anchor.model_copy(update={"origin": AnchorOrigin.DAILY_OBSERVATION})
    return anchor


def _canonicalize_visible_anchor(
    anchor: CaseMetricAnchor,
    text: str,
    profile: SleepProfile,
    insight: LongitudinalInsight,
) -> CaseMetricAnchor:
    """Replace model-rounded metadata with its exact deterministic source value.

    Reader prose may show 88.1% for an exact Profile value of 88.095%.  This repair is
    allowed only when the exact source value renders visibly in the same text and the
    supplied metadata differs by no more than ordinary display rounding.
    """

    anchor = _canonicalize_anchor_origin(anchor)
    resolved = (
        resolve_insight_anchor(insight, anchor)
        if anchor.origin is AnchorOrigin.INSIGHT
        else resolve_profile_anchor(profile, anchor)
    )
    if resolved is None:
        return anchor
    expected_value, expected_unit = resolved
    if (
        anchor.unit == expected_unit
        and abs(anchor.value - expected_value) <= 0.5
        and _anchor_value_is_visible(text, expected_value, expected_unit)
    ):
        return anchor.model_copy(update={"value": expected_value})
    return anchor


def _canonicalize_observation_anchors(
    observation: CaseObservation,
    profile: SleepProfile,
    insight: LongitudinalInsight,
) -> CaseObservation:
    text = _observation_text(observation)
    canonical = [
        _canonicalize_visible_anchor(anchor, text, profile, insight)
        for anchor in observation.anchors
    ]
    return observation.model_copy(
        update={
            "anchors": [
                anchor
                for anchor in canonical
                if _anchor_value_is_visible(text, anchor.value, anchor.unit)
            ]
        }
    )


def _autofill_visible_observation_anchors(
    observation: CaseObservation,
    profile: SleepProfile,
    insight: LongitudinalInsight,
) -> CaseObservation:
    """Recover omitted exact anchors for visible Profile/Insight values.

    In particular, observations often mention the preceding week or a change between
    weeks while the model declares only the recent value.  We recover those declarations
    from deterministic values already present in Profile/Insight; no new value is made up.
    """

    text = _observation_text(observation)
    existing = {
        (
            anchor.origin,
            anchor.metric_id,
            anchor.statistic,
            anchor.related_metric_id,
            anchor.observation_day_index,
            anchor.value,
            anchor.unit,
        )
        for anchor in observation.anchors
    }
    additions: list[CaseMetricAnchor] = []
    candidates = _visible_case_anchor_candidates(
        profile, insight, set(observation.relevant_metric_ids)
    )
    for candidate in candidates:
        identity = (
            candidate.origin,
            candidate.metric_id,
            candidate.statistic,
            candidate.related_metric_id,
            candidate.observation_day_index,
            candidate.value,
            candidate.unit,
        )
        if identity in existing or not _anchor_value_is_visible(text, candidate.value, candidate.unit):
            continue
        additions.append(candidate)
        existing.add(identity)
    return observation.model_copy(update={"anchors": [*observation.anchors, *additions]})


def _visible_case_anchor_candidates(
    profile: SleepProfile,
    insight: LongitudinalInsight,
    metric_ids: set[str],
) -> list[CaseMetricAnchor]:
    """Build exact Profile/Insight anchors used for visible-value recovery."""

    candidates: list[CaseMetricAnchor] = []
    for summary in profile.metric_summaries:
        if summary.metric_id not in metric_ids:
            continue
        for statistic, value in (
            (AnchorStatistic.MEDIAN, summary.median),
            (AnchorStatistic.MEAN, summary.mean),
        ):
            if value is not None:
                candidates.append(
                    CaseMetricAnchor(
                        metric_id=summary.metric_id,
                        origin=AnchorOrigin.AGGREGATE,
                        statistic=statistic,
                        value=value,
                        unit=summary.unit,
                    )
                )
    for item in insight.recent_7_night_comparison:
        if item.metric_id not in metric_ids:
            continue
        for statistic, window in (
            (AnchorStatistic.RECENT_7_CENTER, item.recent_7_nights),
            (AnchorStatistic.PREVIOUS_7_CENTER, item.previous_7_nights),
            (AnchorStatistic.EARLIER_HISTORY_CENTER, item.earlier_history),
        ):
            if window is not None:
                candidates.append(
                    CaseMetricAnchor(
                        metric_id=item.metric_id,
                        origin=AnchorOrigin.INSIGHT,
                        statistic=statistic,
                        value=window.center,
                        unit=item.unit,
                    )
                )
        for statistic, value in (
            (AnchorStatistic.RECENT_MINUS_PREVIOUS_7, item.recent_minus_previous),
            (AnchorStatistic.RECENT_MINUS_EARLIER_HISTORY, item.recent_minus_earlier_history),
        ):
            if value is None:
                continue
            unit = (
                "min"
                if item.unit == "clock_minute"
                and statistic
                in {
                    AnchorStatistic.RECENT_MINUS_PREVIOUS_7,
                    AnchorStatistic.RECENT_MINUS_EARLIER_HISTORY,
                }
                else item.unit
            )
            candidates.append(
                CaseMetricAnchor(
                    metric_id=item.metric_id,
                    origin=AnchorOrigin.INSIGHT,
                    statistic=statistic,
                    value=value,
                    unit=unit,
                )
            )
    return candidates


def _autofill_visible_draft_anchors(
    draft: RecommendationDraft,
    profile: SleepProfile,
    insight: LongitudinalInsight,
    observation: CaseObservation | None = None,
) -> RecommendationDraft:
    """Recover exact case anchors for visible derived comparisons in a draft."""

    updates: dict[str, SectionGrounding] = {}
    observed_metrics = set(observation.relevant_metric_ids) if observation is not None else set()
    for section in ("section_1", "section_2", "section_3", "section_4"):
        grounding = getattr(draft, f"{section}_grounding")
        metric_ids = observed_metrics | {
            anchor.metric_id for anchor in grounding.case_anchors
        } | set(grounding.outcome_metrics) | {
            reference.metric_id for reference in grounding.evidence_reference_values
        }
        if not metric_ids:
            continue
        text = getattr(draft, section)
        candidates = _visible_case_anchor_candidates(profile, insight, metric_ids)
        retained = list(grounding.case_anchors)
        existing = {
            (
                anchor.origin,
                anchor.metric_id,
                anchor.statistic,
                anchor.related_metric_id,
                anchor.observation_day_index,
                anchor.value,
                anchor.unit,
            )
            for anchor in retained
        }
        for span in _insight_numeric_spans(text):
            if _is_fixed_comparison_period(span.text_span):
                continue
            if any(_numeric_span_matches_anchor(span, anchor) for anchor in retained):
                continue
            matches = [
                candidate
                for candidate in candidates
                if _numeric_span_matches_anchor(span, candidate)
            ]
            if re.search(
                r"\b(?:fewer|more|less|additional|extra|longer|shorter|later|earlier)\b",
                span.context,
                re.I,
            ):
                matches.sort(
                    key=lambda item: item.statistic
                    not in {
                        AnchorStatistic.RECENT_MINUS_PREVIOUS_7,
                        AnchorStatistic.RECENT_MINUS_EARLIER_HISTORY,
                    }
                )
            if matches:
                candidate = matches[0]
                identity = (
                    candidate.origin,
                    candidate.metric_id,
                    candidate.statistic,
                    candidate.related_metric_id,
                    candidate.observation_day_index,
                    candidate.value,
                    candidate.unit,
                )
                if identity not in existing:
                    retained.append(candidate)
                    existing.add(identity)
        updates[f"{section}_grounding"] = grounding.model_copy(
            update={"case_anchors": retained}
        )
    return draft.model_copy(update=updates)


def _canonicalize_draft_anchors(
    draft: RecommendationDraft,
    profile: SleepProfile,
    insight: LongitudinalInsight,
) -> RecommendationDraft:
    updates: dict[str, object] = {}
    for section in ("section_1", "section_2", "section_3", "section_4"):
        grounding = getattr(draft, f"{section}_grounding")
        updates[f"{section}_grounding"] = grounding.model_copy(
            update={
                "case_anchors": [
                    _canonicalize_visible_anchor(
                        anchor, getattr(draft, section), profile, insight
                    )
                    for anchor in grounding.case_anchors
                ]
            }
        )
    return draft.model_copy(update=updates)


def _autofill_visible_evidence_references(
    draft: RecommendationDraft,
    library: EvidenceCardLibrary,
) -> RecommendationDraft:
    """Complete audit references when the Composer visibly used a selected card value.

    Reference anchors are metadata, not reader text. The Composer can correctly state a
    population comparison while omitting its structured declaration; recovering that
    declaration from the selected card avoids retrying an otherwise valid draft.
    """

    by_key = {card.evidence_key: card for card in library.cards}
    updates: dict[str, object] = {}
    for section in ("section_1", "section_2", "section_3", "section_4"):
        text = getattr(draft, section)
        grounding = getattr(draft, f"{section}_grounding")
        existing = {
            (
                item.evidence_key,
                item.metric_id,
                item.comparator,
                item.value,
                item.unit,
                item.label,
            )
            for item in grounding.evidence_reference_values
        }
        additions: list[EvidenceReferenceAnchor] = []
        for key in grounding.evidence_keys:
            card = by_key.get(key)
            if card is None:
                continue
            for value in card.reference_values:
                identity = (
                    key,
                    value.metric_id,
                    value.comparator,
                    value.value,
                    value.unit,
                    value.label,
                )
                if identity in existing or not _anchor_value_is_visible(
                    text, value.value, value.unit
                ):
                    continue
                additions.append(
                    EvidenceReferenceAnchor(
                        evidence_key=key,
                        metric_id=value.metric_id,
                        comparator=value.comparator,
                        value=value.value,
                        unit=value.unit,
                        label=value.label,
                    )
                )
                existing.add(identity)
        if additions:
            updates[f"{section}_grounding"] = grounding.model_copy(
                update={
                    "evidence_reference_values": [
                        *grounding.evidence_reference_values,
                        *additions,
                    ]
                }
            )
    return draft.model_copy(update=updates)


def _global_rules(path: str | Path | None = None) -> list[dict]:
    root = Path(__file__).resolve().parents[3]
    rule_path = (
        Path(path)
        if path is not None
        else root / "evidence_bank" / "interpretation_rules.json"
    )
    return json.loads(rule_path.read_text(encoding="utf-8"))["rules"]


def _applicable_global_rules(
    observation: CaseObservation,
    selection: EvidenceSelection,
) -> list[dict]:
    relevant_metrics = set(observation.relevant_metric_ids)
    relevant_metrics.update(
        metric for item in selection.selected for metric in item.matched_outcomes
    )
    return [
        {"rule_id": item["rule_id"], "instruction": item["instruction"]}
        for item in _global_rules()
        if item.get("scope") == "universal"
        or relevant_metrics.intersection(item.get("applies_when_any_metric") or [])
    ]


def _observation_text(observation: CaseObservation) -> str:
    return "\n".join(
        [
            observation.primary_observation,
            *observation.supporting_observations,
            *([observation.context_observation] if observation.context_observation else []),
        ]
    )


def _is_fixed_comparison_period(span: str) -> bool:
    """Return whether a numeric span names the pipeline's fixed 7-night view."""

    return bool(re.fullmatch(r"(?:the\s+)?7\s+nights?", span.strip(), re.I))


def _numeric_span_matches_anchor(span: object, anchor: CaseMetricAnchor) -> bool:
    """Match a parsed span to one exact case anchor without cross-sentence matching."""

    span_value = float(getattr(span, "value"))
    span_unit = str(getattr(span, "unit"))
    if span_unit == anchor.unit:
        if span_unit in {"min", "minute", "minutes"}:
            same_display_value = round(abs(span_value)) == round(abs(anchor.value))
        elif span_unit == "%":
            same_display_value = round(abs(span_value), 1) == round(abs(anchor.value), 1)
        else:
            same_display_value = abs(abs(span_value) - abs(anchor.value)) <= 1e-6
        return (
            same_display_value
            and _anchor_value_is_visible(getattr(span, "text_span"), anchor.value, anchor.unit)
        )
    # A bare number such as the ``10`` in ``10 fewer minutes`` needs its sentence
    # context to recover the unit, but the numeric value disambiguates other anchors.
    return (
        span_unit == "unitless"
        and round(abs(span_value)) == round(abs(anchor.value))
        and _anchor_value_is_visible(getattr(span, "context"), anchor.value, anchor.unit)
    )


def validate_observation(
    observation: CaseObservation,
    profile: SleepProfile,
    insight: LongitudinalInsight,
) -> list[str]:
    issues: list[str] = []
    if observation.case_id != profile.case_id:
        issues.append("Observation case_id differs from Profile")
    available = {item.metric_id for item in profile.metric_summaries}
    unavailable = sorted(set(observation.relevant_metric_ids) - available)
    if unavailable:
        issues.append(f"Observation names unavailable metrics: {unavailable}")
    text = _observation_text(observation)
    observation_fields = [
        observation.primary_observation,
        *observation.supporting_observations,
        *([observation.context_observation] if observation.context_observation else []),
    ]
    normalized_fields = [
        re.sub(r"\s+", " ", item.strip()).casefold() for item in observation_fields
    ]
    if len(normalized_fields) != len(set(normalized_fields)):
        issues.append("Observation repeats the same fact in more than one field")
    for field in observation_fields:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", field.strip())
        if len(sentences) > 1:
            issues.append(
                "Each Observation field must contain one focused sentence; move a distinct "
                "fact to supporting_observations or remove it"
            )
            break
    if "?" in text:
        issues.append("Observation must state recorded facts, not an interpretive question")
    if re.search(r"\b(?:main|primary) (?:concern|issue|problem)\b", text, re.I):
        issues.append(
            "Observation labels a recorded pattern as a main concern or problem; state the "
            "metric comparison only and leave prioritization to Evidence and Pattern Analysis"
        )
    for anchor in observation.anchors:
        resolved = (
            resolve_insight_anchor(insight, anchor)
            if anchor.origin is AnchorOrigin.INSIGHT
            else resolve_profile_anchor(profile, anchor)
        )
        if resolved is None:
            issues.append(
                "Observation has an unresolved anchor: "
                f"{anchor.origin.value}/{anchor.metric_id}/{anchor.statistic.value}"
            )
            continue
        expected_value, expected_unit = resolved
        if anchor.unit != expected_unit or abs(anchor.value - expected_value) > 1e-6:
            issues.append(
                "Observation anchor differs from Profile/Insight: "
                f"{anchor.origin.value}/{anchor.metric_id}/{anchor.statistic.value}"
            )
        elif not _anchor_value_is_visible(text, expected_value, expected_unit):
            issues.append(
                "Observation anchor is not visible in its prose: "
                f"{anchor.origin.value}/{anchor.metric_id}/{anchor.statistic.value}"
            )
    for span in _insight_numeric_spans(text):
        if _is_fixed_comparison_period(span.text_span):
            continue
        if not any(_numeric_span_matches_anchor(span, anchor) for anchor in observation.anchors):
            # A phrase such as "1 hour 19 minutes later" is a derived difference
            # between two recorded values and has no single source anchor. The paired
            # source values remain available in Profile/Insight for audit and Stage 2.
            if re.search(
                r"\b(?:later|earlier|more|less|shorter|longer|difference|change|"
                r"increased|decreased|up|down)\b",
                span.context,
                re.I,
            ):
                continue
            issues.append(f"Observation numeric span {span.text_span!r} has no declared anchor")
    for label, pattern in _INTERNAL_READER_LANGUAGE.items():
        if re.search(pattern, text, re.I):
            issues.append(_language_issue("Observation", label))
    for label, pattern in _INTERMEDIATE_INTERNAL_LANGUAGE.items():
        if re.search(pattern, text, re.I):
            issues.append(f"Observation contains {label}")
    if re.search(r"\b(?:IQR|observation day)\b", text, re.I):
        issues.append("Observation contains internal statistical or indexing language")
    return issues


def validate_selection(
    selection: EvidenceSelection,
    observation: CaseObservation,
    library: EvidenceCardLibrary,
    *,
    allowed_keys: set[str] | None = None,
    profile: SleepProfile | None = None,
    insight: LongitudinalInsight | None = None,
) -> list[str]:
    issues: list[str] = []
    if selection.case_id != observation.case_id:
        issues.append("Evidence Selection case_id differs from Observation")
    selection_text = " ".join(
        [selection.selection_summary]
        + [item.case_relevance for item in selection.selected]
    )
    for label, pattern in _INTERMEDIATE_INTERNAL_LANGUAGE.items():
        if re.search(pattern, selection_text, re.I):
            issues.append(
                f"Evidence Selection contains {label}; explain why the knowledge applies "
                "to the sleep pattern without naming the pipeline"
            )
    for label, pattern in _INTERNAL_READER_LANGUAGE.items():
        if re.search(pattern, selection_text, re.I):
            issues.append(_language_issue("Evidence Selection", label))
    by_key = {card.evidence_key: card for card in library.cards if card.runtime_enabled}
    have_candidate_context = profile is not None and insight is not None
    candidate_keys = (
        {
            card.evidence_key
            for card in candidate_evidence_pools(profile, insight, library).all
        }
        if profile is not None and insight is not None
        else set()
    )
    inapplicable_gated = sorted(
        {
            item.evidence_key
            for item in selection.selected
            if item.evidence_key in _DETERMINISTIC_APPLICABILITY_KEYS
            and have_candidate_context
            and item.evidence_key not in candidate_keys
        }
    )
    if inapplicable_gated:
        issues.append(
            "Evidence Selection contains reference or clinical-boundary cards whose "
            f"deterministic applicability is not met: {inapplicable_gated}"
        )
    unknown = sorted({item.evidence_key for item in selection.selected} - set(by_key))
    if unknown:
        issues.append(f"Evidence Selection contains unknown or disabled keys: {unknown}")
    outside_candidates = (
        sorted({item.evidence_key for item in selection.selected} - allowed_keys)
        if allowed_keys is not None
        else []
    )
    if outside_candidates:
        issues.append(
            "Evidence Selection contains keys outside the retrieved candidates: "
            f"{outside_candidates}"
        )
    conditional = [
        item.evidence_key
        for item in selection.selected
        if item.evidence_key in by_key and by_key[item.evidence_key].condition
    ]
    if len(conditional) > 1:
        issues.append("Evidence Selection contains more than one unobserved condition")
    selected_cards = [by_key[item.evidence_key] for item in selection.selected if item.evidence_key in by_key]
    if not any(card.next_step for card in selected_cards):
        issues.append(
            "Evidence Selection must include one card that legally supports the primary "
            "Personalized Recommendation; select direct guidance, maintenance, or a justified "
            "conditional boundary."
        )
    for item in selection.selected:
        card = by_key.get(item.evidence_key)
        if card is None:
            continue
        unsupported = sorted(set(item.matched_outcomes) - set(card.evaluate_with))
        if unsupported:
            issues.append(f"{item.evidence_key} has unsupported matched outcomes: {unsupported}")

    caffeine_key = "caffeine-close-to-sleep-may-disrupt-sleep"
    caffeine_item = next(
        (item for item in selection.selected if item.evidence_key == caffeine_key), None
    )
    if caffeine_item is not None:
        matched = set(caffeine_item.matched_outcomes)
        continuity = {"waso_min", "sleep_efficiency_pct"}
        if "total_sleep_min" not in matched or not matched.intersection(continuity):
            issues.append(
                "Caffeine guidance requires matched repeated short sleep and reduced continuity"
            )
        if insight is not None:
            recent_sleep_item = next(
                (
                    item
                    for item in insight.recent_7_night_comparison
                    if item.metric_id == "total_sleep_min"
                ),
                None,
            )
            recent_tib_item = next(
                (
                    item
                    for item in insight.recent_7_night_comparison
                    if item.metric_id == "time_in_bed_min"
                ),
                None,
            )
            recent_sleep = (
                recent_sleep_item.recent_7_nights.center
                if recent_sleep_item is not None
                and recent_sleep_item.recent_7_nights is not None
                else None
            )
            recent_tib = (
                recent_tib_item.recent_7_nights.center
                if recent_tib_item is not None
                and recent_tib_item.recent_7_nights is not None
                else None
            )
            if recent_sleep is None or recent_sleep >= 419.0:
                issues.append(
                    "Caffeine guidance requires repeated recent sleep clearly below 7 hours"
                )
            if recent_tib is None or recent_tib < 419.0:
                issues.append(
                    "Caffeine guidance is not applicable when limited recent time in bed "
                    "is the more direct short-sleep constraint"
                )

    allow_more_key = "allow-more-time-for-sleep"
    if any(item.evidence_key == allow_more_key for item in selection.selected):
        recent_sleep = None
        recent_tib = None
        if insight is not None:
            recent_sleep_item = next(
                (
                    item
                    for item in insight.recent_7_night_comparison
                    if item.metric_id == "total_sleep_min"
                ),
                None,
            )
            recent_tib_item = next(
                (
                    item
                    for item in insight.recent_7_night_comparison
                    if item.metric_id == "time_in_bed_min"
                ),
                None,
            )
            recent_sleep = (
                recent_sleep_item.recent_7_nights.center
                if recent_sleep_item is not None
                and recent_sleep_item.recent_7_nights is not None
                else None
            )
            recent_tib = (
                recent_tib_item.recent_7_nights.center
                if recent_tib_item is not None
                and recent_tib_item.recent_7_nights is not None
                else None
            )
        weekday_sleep = None
        weekday_tib = None
        if profile is not None:
            sleep_weekpart = next(
                (
                    item
                    for item in profile.weekday_weekend_comparisons
                    if item.metric_id == "total_sleep_min" and item.reportable
                ),
                None,
            )
            tib_weekpart = next(
                (
                    item
                    for item in profile.weekday_weekend_comparisons
                    if item.metric_id == "time_in_bed_min" and item.reportable
                ),
                None,
            )
            weekday_sleep = sleep_weekpart.weekday_center if sleep_weekpart else None
            weekday_tib = tib_weekpart.weekday_center if tib_weekpart else None
        recent_limited = bool(
            recent_sleep is not None
            and recent_tib is not None
            and recent_sleep < 419.0
            and recent_tib < 419.0
        )
        weekday_limited = bool(
            weekday_sleep is not None
            and weekday_tib is not None
            and weekday_sleep < 419.0
            and weekday_tib < 419.0
        )
        if not recent_limited and not weekday_limited:
            issues.append(
                "Allow-more-time guidance requires short sleep and less than 7 hours in "
                "bed in the same recent or weekday pattern"
            )

    activity_key = "regular-daytime-movement-may-support-sleep-efficiency"
    if any(item.evidence_key == activity_key for item in selection.selected):
        activity_association = None
        if profile is not None:
            activity_association = next(
                (
                    item
                    for item in profile.daily_context_associations
                    if item.context_metric_id == "active_minutes"
                    and item.sleep_metric_id == "sleep_efficiency_pct"
                    and item.reportable
                    and item.high_minus_low_sleep is not None
                    and item.high_minus_low_sleep > 0
                ),
                None,
            )
        if activity_association is None:
            issues.append(
                "Daytime-movement guidance requires a reportable positive within-person "
                "activity-to-sleep-efficiency association"
            )
    return issues


def plan_case(
    client: LLMClient,
    profile: SleepProfile,
    insight: LongitudinalInsight,
    library: EvidenceCardLibrary,
    *,
    previous_plan: Stage1Plan | None = None,
    review_feedback: list[dict] | None = None,
) -> tuple[Stage1Plan, list[str]]:
    """Generate O, then E, in one call over the complete runtime Evidence bank."""

    runtime_cards = [card for card in library.cards if card.runtime_enabled]
    candidate_keys = {
        card.evidence_key
        for card in candidate_evidence_pools(profile, insight, library).all
    }
    plan = client.structured(
        prompt=read_prompt("stage1/evidence_agent"),
        payload={
            "evidence_release_id": library.release_id,
            "generation_evidence_bank": [writer_card(card) for card in runtime_cards],
            "deterministic_applicability": {
                "gated_evidence_keys": sorted(_DETERMINISTIC_APPLICABILITY_KEYS),
                "applicable_evidence_keys": sorted(
                    candidate_keys.intersection(_DETERMINISTIC_APPLICABILITY_KEYS)
                ),
                "rule": (
                    "A gated key may be selected only when it appears in "
                    "applicable_evidence_keys. This checks measurable prerequisites; it "
                    "does not select the final minimal Evidence set."
                ),
            },
            **generation_model_input(
                profile.model_dump(mode="json"), insight.model_dump(mode="json")
            ),
            **(
                {"previous_plan": previous_plan.model_dump(mode="json")}
                if previous_plan is not None
                else {}
            ),
            **({"review_feedback": review_feedback} if review_feedback else {}),
        },
        schema=Stage1Plan,
    )
    plan = plan.model_copy(
        update={
            "observation": _canonicalize_observation_anchors(
                _autofill_visible_observation_anchors(plan.observation, profile, insight),
                profile,
                insight,
            )
        }
    )
    issues = validate_observation(plan.observation, profile, insight)
    issues.extend(
        validate_selection(
            plan.evidence_selection,
            plan.observation,
            library,
            profile=profile,
            insight=insight,
        )
    )
    case_signals = generation_case_signals(profile.model_dump(mode="json"))
    selected_keys = {
        item.evidence_key for item in plan.evidence_selection.selected
    }
    if (
        case_signals["weekday_weekend_whole_sleep_period_shift_is_midpoint_decisive"]
        and selected_keys.intersection(_TIMING_EVIDENCE_KEYS)
    ):
        if "sleep_midpoint_clock_min" not in set(
            plan.observation.relevant_metric_ids
        ):
            issues.append(
                "Observation must retain sleep_midpoint_clock_min when the middle of the "
                "sleep period is the decisive weekday/weekend timing signal"
            )
        if not _NATURAL_WHOLE_SLEEP_PERIOD.search(_observation_text(plan.observation)):
            issues.append(
                "Observation must say naturally that the middle or whole sleep period "
                "shifted on weekends when midpoint is the decisive timing signal"
            )
    return plan, issues


def validate_recommendation(
    draft: RecommendationDraft,
    profile: SleepProfile,
    insight: LongitudinalInsight,
    selection: EvidenceSelection,
    library: EvidenceCardLibrary,
) -> list[str]:
    issues: list[str] = []
    if draft.case_id != profile.case_id:
        issues.append("Recommendation case_id differs from Profile")
    selected = {item.evidence_key for item in selection.selected}
    groundings = {
        section: getattr(draft, f"{section}_grounding")
        for section in ("section_1", "section_2", "section_3", "section_4")
    }
    used = {key for grounding in groundings.values() for key in grounding.evidence_keys}
    if used != selected:
        issues.append(
            "Final used Evidence must equal the minimal selected set: "
            f"selected={sorted(selected)}, used={sorted(used)}"
        )
    by_key = {card.evidence_key: card for card in library.cards}
    visible = "\n".join(draft.render_sections().values())
    leaked_evidence_keys = sorted(
        key for key in by_key if key.lower() in visible.lower()
    )
    if leaked_evidence_keys:
        issues.append(
            "Reader text exposes internal Evidence keys: "
            f"{leaked_evidence_keys}"
        )
    unsupported_used = sorted(used - set(by_key))
    if unsupported_used:
        issues.append(f"Recommendation uses unknown Evidence: {unsupported_used}")
    section_3_cards = [
        by_key[key] for key in groundings["section_3"].evidence_keys if key in by_key
    ]
    if not any(card.next_step for card in section_3_cards):
        issues.append("Section 3 has no Evidence card that permits a next step")
    selected_action_keys = {
        key for key in selected if key in by_key and by_key[key].next_step
    }
    section_3_action_cards = [card for card in section_3_cards if card.next_step]
    section_3_action_keys = {card.evidence_key for card in section_3_action_cards}
    if section_3_action_keys != selected_action_keys:
        issues.append(
            "Section 3 must contain one action item for every selected action-permitting "
            f"Evidence card: selected={sorted(selected_action_keys)}, "
            f"section_3={sorted(section_3_action_keys)}"
        )
    action_titles = _ACTION_TITLE.findall(draft.section_3)
    if len(action_titles) != len(section_3_action_cards):
        issues.append(
            "Section 3 must contain exactly one bold controlled title per action item: "
            f"titles={len(action_titles)}, actions={len(section_3_action_cards)}"
        )
    if len(action_titles) != len(set(action_titles)):
        issues.append("Section 3 repeats a controlled action title")
    allowed_titles = {
        title for card in section_3_action_cards for title in card.allowed_action_titles
    }
    unapproved_titles = sorted(set(action_titles) - allowed_titles)
    if unapproved_titles:
        issues.append(
            f"Section 3 contains titles outside allowed_action_titles: {unapproved_titles}"
        )
    for card in section_3_action_cards:
        matching = [title for title in action_titles if title in card.allowed_action_titles]
        if len(matching) != 1:
            issues.append(
                f"Section 3 must use exactly one controlled title for {card.evidence_key!r}"
            )
    for card in section_3_cards:
        if card.condition and not _EXPLICIT_CONDITION_LANGUAGE.search(draft.section_3):
            issues.append(
                f"Section 3 uses conditional Evidence {card.evidence_key!r} without "
                "explicit conditional language"
            )
    if _TIMING_CLINICAL_EVIDENCE_KEY in selected:
        if not re.search(r"\bmelatonin\b", draft.section_3, re.I):
            issues.append(
                "Section 3 must name melatonin as a professional-guidance boundary when "
                "unwanted timing clinical review is selected"
            )
        if not re.search(r"\b(?:doctor|clinician|sleep specialist)\b", draft.section_3, re.I):
            issues.append(
                "Section 3 must name a doctor, clinician, or sleep specialist when unwanted "
                "timing clinical review is selected"
            )
        if re.search(r"\bmelatonin\b", draft.section_4, re.I):
            issues.append(
                "Section 4 repeats the melatonin guardrail from Personalized Recommendation; evaluate "
                "whether the unwanted disruptive timing persists and when to discuss it"
            )
        if re.search(
            r"\b(?:energy|concentration|sleepiness|unrefreshed|unrefreshing)\b",
            f"{draft.section_3}\n{draft.section_4}",
            re.I,
        ):
            issues.append(
                "The unwanted-timing clinical boundary supports preference, persistence, "
                "and interference with necessary responsibilities; do not add unrecorded "
                "energy, concentration, sleepiness, or refreshing-sleep symptoms"
            )
    elif re.search(r"\bmelatonin\b", visible, re.I):
        issues.append(
            "Reader text mentions melatonin without the selected unwanted-timing clinical boundary"
        )
    if (
        _MAINTENANCE_EVIDENCE_KEY in groundings["section_3"].evidence_keys
        and re.search(r"\bmain (?:issue|concern|problem)\b", draft.section_2, re.I)
    ):
        issues.append(
            "Maintenance is the primary action, so Section 2 must foreground the adequate "
            "or stable dimension and keep any unresolved secondary measure in context"
        )
    allowed_outcomes = {
        metric
        for key in groundings["section_4"].evidence_keys
        if key in by_key
        for metric in by_key[key].evaluate_with
    }
    invalid_outcomes = sorted(set(groundings["section_4"].outcome_metrics) - allowed_outcomes)
    if invalid_outcomes:
        issues.append(f"Section 4 evaluates unsupported outcomes: {invalid_outcomes}")
    clinical_discussion_is_primary = bool(section_3_cards) and all(
        card.kind is EvidenceKind.BOUNDARY for card in section_3_cards
    )
    if not groundings["section_4"].outcome_metrics and not clinical_discussion_is_primary:
        issues.append("Section 4 must name at least one grounded evaluation outcome")

    selected_clinical_boundary = any(
        key in by_key
        and by_key[key].kind is EvidenceKind.BOUNDARY
        and by_key[key].next_step
        and re.search(r"\b(?:clinician|clinical|sleep specialist)\b", by_key[key].next_step, re.I)
        for key in selected
    )
    if re.search(r"\bdaytime (?:problem|problems|effect|effects|symptom|symptoms)\b", visible, re.I):
        if not selected_clinical_boundary:
            issues.append(
                "reader text adds daytime symptoms without selected clinical-boundary Evidence"
            )

    case_signals = generation_case_signals(profile.model_dump(mode="json"))
    if (
        case_signals["weekday_weekend_whole_sleep_period_shift_is_midpoint_decisive"]
        and selected.intersection(_TIMING_EVIDENCE_KEYS)
        and not _NATURAL_WHOLE_SLEEP_PERIOD.search(visible)
    ):
        issues.append(
            "Reader text must describe the middle or whole sleep period shift naturally "
            "when it is the decisive timing signal"
        )
    for key, term_pattern in _UNOBSERVED_CONDITION_TERMS.items():
        if key in selected and term_pattern.search(draft.section_2):
            issues.append(
                "Section 2 repeats an unobserved conditional factor; interpret only the "
                "recorded sleep pattern there and introduce the factor once in Section 3"
            )
    if selected.intersection(_TIMING_EVIDENCE_KEYS):
        timing_follow_up_terms = (
            re.search(r"\bsleep[- ]start\b", draft.section_4, re.I),
            re.search(r"\bwake(?: time| timing)?\b", draft.section_4, re.I),
            re.search(r"\bmiddle of (?:your|the) sleep period\b", draft.section_4, re.I),
        )
        if all(timing_follow_up_terms):
            issues.append(
                "Section 4 lists three timing outcomes; evaluate the action's timing "
                "component or whether weekday and weekend sleep times move closer together"
            )

    for section, grounding in groundings.items():
        text = getattr(draft, section)
        duplicate_keys = len(grounding.evidence_keys) != len(set(grounding.evidence_keys))
        if duplicate_keys:
            issues.append(f"{section} has duplicate Evidence keys")
        for anchor in grounding.case_anchors:
            resolved = (
                resolve_insight_anchor(insight, anchor)
                if anchor.origin is AnchorOrigin.INSIGHT
                else resolve_profile_anchor(profile, anchor)
            )
            if resolved is None:
                issues.append(
                    f"{section} has an unresolved anchor: "
                    f"{anchor.origin.value}/{anchor.metric_id}/{anchor.statistic.value}"
                )
                continue
            expected_value, expected_unit = resolved
            if anchor.unit != expected_unit or abs(anchor.value - expected_value) > 1e-6:
                issues.append(f"{section} anchor differs from Profile/Insight")
            elif not _anchor_value_is_visible(text, expected_value, expected_unit):
                issues.append(f"{section} contains an invisible or unused anchor")
        for reference in grounding.evidence_reference_values:
            key = reference.evidence_key
            card = by_key.get(key)
            declared = card is not None and any(
                value.metric_id == reference.metric_id
                and value.comparator == reference.comparator
                and value.unit == reference.unit
                and value.label == reference.label
                and abs(value.value - reference.value) <= 1e-9
                for value in card.reference_values
            )
            if key not in grounding.evidence_keys:
                issues.append(f"{section} reference Evidence is not grounded in the Section")
            elif not declared:
                issues.append(f"{section} contains an undeclared Evidence reference value")
            elif not _anchor_value_is_visible(text, reference.value, reference.unit):
                issues.append(f"{section} Evidence reference value is not visible")
        for span in _insight_numeric_spans(text):
            if _is_fixed_comparison_period(span.text_span):
                continue
            case_supported = any(
                _numeric_span_matches_anchor(span, anchor)
                for anchor in grounding.case_anchors
            )
            evidence_supported = any(
                _anchor_value_is_visible(span.text_span, reference.value, reference.unit)
                for reference in grounding.evidence_reference_values
            )
            if not case_supported and not evidence_supported:
                issues.append(
                    f"{section} numeric span {span.text_span!r} has no case or Evidence anchor"
                )

    for section in ("section_1", "section_2", "section_3", "section_4"):
        section_text = getattr(draft, section)
        for label, pattern in _INTERNAL_READER_LANGUAGE.items():
            if re.search(pattern, section_text, re.I):
                issues.append(_language_issue(section, label))
        for label in _reader_sentence_style_issues(section_text):
            issues.append(_language_issue(section, label))
    if re.search(r"\bmay not precisely represent\b", draft.section_1, re.I):
        issues.append("section_1 contains a measurement limitation that belongs in Pattern Analysis")
    if re.search(r"\b(?:limited|reduced) sleep continuity\b", draft.section_1, re.I):
        issues.append(
            "section_1 uses an abstract continuity label instead of the observable sleep measure"
        )
    if re.search(
        r"\bto (?:support|address) (?:the )?(?:remaining )?"
        r"(?:sleep )?(?:continuity concern|pattern)\b",
        draft.section_3,
        re.I,
    ):
        issues.append(
            "Section 3 repeats Pattern Analysis with an abstract rationale; keep the action direct"
        )
    activity_key = "regular-daytime-movement-may-support-sleep-efficiency"
    if activity_key in selected and not re.search(
        r"\b(?:active|activity|movement)\b", draft.section_2, re.I
    ):
        issues.append(
            "Section 2 must state the case-specific activity-to-sleep-efficiency "
            "association before Section 3 recommends daytime movement"
        )
    if activity_key in selected and re.search(
        r"\b(?:activity|movement)\s+(?:is|was|appears to be)\s+helping\b|"
        r"\bhelping your sleep efficiency\b",
        draft.section_4,
        re.I,
    ):
        issues.append(
            "Section 4 turns the observed activity-efficiency association into a causal "
            "benefit; say only whether the same personal pattern continues"
        )
    for section, grounding in groundings.items():
        if re.search(r"\baveraged?\b", getattr(draft, section), re.I) and not any(
            anchor.statistic is AnchorStatistic.MEAN for anchor in grounding.case_anchors
        ):
            issues.append(f"{section} uses average language without a mean anchor")
    return issues


def write_recommendation(
    client: LLMClient,
    profile: SleepProfile,
    insight: LongitudinalInsight,
    observation: CaseObservation,
    selection: EvidenceSelection,
    library: EvidenceCardLibrary,
    *,
    previous_draft: RecommendationDraft | None = None,
    review_feedback: list[dict] | None = None,
    editable_sections: set[str] | None = None,
) -> tuple[RecommendationDraft, list[str]]:
    case_evidence = build_case_evidence(selection, library)
    prompt_names = [
        "shared/guidance_contract",
        "stage1/composer_agent",
        "shared/reader_language_contract",
    ]
    if previous_draft is not None and review_feedback:
        prompt_names.append("stage1/composer_revision")
    draft = client.structured(
        prompt=read_prompt_bundle(*prompt_names),
        payload={
            "evidence_release_id": library.release_id,
            "global_interpretation_rules": _applicable_global_rules(observation, selection),
            "case_evidence": case_evidence,
            "observation": observation.model_dump(mode="json", exclude={"anchors"}),
            "available_case_anchors": [
                anchor.model_dump(mode="json") for anchor in observation.anchors
            ],
            **generation_model_input(
                profile.model_dump(mode="json"), insight.model_dump(mode="json")
            ),
            **(
                {"previous_draft": previous_draft.model_dump(mode="json")}
                if previous_draft is not None
                else {}
            ),
            **({"review_feedback": review_feedback} if review_feedback else {}),
        },
        schema=RecommendationDraft,
    )
    draft = _canonicalize_draft_anchors(draft, profile, insight)
    draft = _autofill_visible_draft_anchors(draft, profile, insight, observation)
    draft = _autofill_visible_evidence_references(draft, library)
    if previous_draft is not None and editable_sections:
        updates: dict[str, object] = {"case_id": previous_draft.case_id}
        for section in ("section_1", "section_2", "section_3", "section_4"):
            if section in editable_sections:
                continue
            updates[section] = getattr(previous_draft, section)
            updates[f"{section}_grounding"] = getattr(previous_draft, f"{section}_grounding")
        draft = draft.model_copy(update=updates)
    # Grounding is audit metadata. Silently remove declarations that the Composer did not
    # actually use; visible unsupported numbers are still rejected below.
    cleaned: dict[str, object] = {}
    for section in ("section_1", "section_2", "section_3", "section_4"):
        text = getattr(draft, section)
        grounding = getattr(draft, f"{section}_grounding")
        cleaned[f"{section}_grounding"] = grounding.model_copy(
            update={
                "case_anchors": [
                    anchor
                    for anchor in grounding.case_anchors
                    if _anchor_value_is_visible(text, anchor.value, anchor.unit)
                ],
                "evidence_reference_values": [
                    reference
                    for reference in grounding.evidence_reference_values
                    if _anchor_value_is_visible(text, reference.value, reference.unit)
                ],
            }
        )
    draft = draft.model_copy(update=cleaned)
    issues = validate_recommendation(draft, profile, insight, selection, library)
    return draft, issues


def polish_recommendation(
    client: LLMClient,
    profile: SleepProfile,
    insight: LongitudinalInsight,
    selection: EvidenceSelection,
    library: EvidenceCardLibrary,
    source_draft: RecommendationDraft,
    *,
    previous_polished_draft: RecommendationDraft | None = None,
    review_feedback: list[dict] | None = None,
) -> tuple[RecommendationDraft, list[str]]:
    """Polish visible prose after reasoning is fixed, then revalidate the full contract."""

    polished = client.structured(
        prompt=read_prompt("stage1/composer_revision"),
        payload={
            "case_id": source_draft.case_id,
            "source_draft": source_draft.model_dump(mode="json"),
            **(
                {
                    "previous_polished_draft": previous_polished_draft.model_dump(
                        mode="json"
                    )
                }
                if previous_polished_draft is not None
                else {}
            ),
            **({"review_feedback": review_feedback} if review_feedback else {}),
        },
        schema=RecommendationDraft,
    )
    polished = _canonicalize_draft_anchors(polished, profile, insight)
    polished = _autofill_visible_evidence_references(polished, library)
    source_titles = _ACTION_TITLE.findall(source_draft.section_3)
    polished_titles = _ACTION_TITLE.findall(polished.section_3)
    if len(source_titles) == len(polished_titles):
        title_iter = iter(source_titles)
        restored_section_3 = _ACTION_TITLE.sub(
            lambda _: f"**{next(title_iter)}**", polished.section_3
        )
    else:
        restored_section_3 = polished.section_3
    immutable_updates: dict[str, object] = {
        "case_id": source_draft.case_id,
        "section_3": restored_section_3,
    }
    for section in ("section_1", "section_2", "section_3", "section_4"):
        immutable_updates[f"{section}_grounding"] = getattr(
            source_draft, f"{section}_grounding"
        )
    polished = polished.model_copy(update=immutable_updates)
    issues: list[str] = []
    issues.extend(
        validate_recommendation(polished, profile, insight, selection, library)
    )
    return polished, list(dict.fromkeys(issues))


def _normalize_evidence_only_review(
    review: BlindReverseReview,
    *,
    case_id: str,
    recommendation: ReaderRecommendation,
) -> BlindReverseReview:
    """Discard out-of-scope review opinions and canonicalize copied reader text."""

    normalized_sections = {}
    for section, result in review.sections().items():
        evidence_issues = [
            issue
            for issue in result.issues
            if issue.issue_type is ReverseIssueType.EVIDENCE
        ]
        normalized_sections[section] = result.model_copy(
            update={
                "section_text": getattr(recommendation, section),
                "issues": evidence_issues,
            }
        )
    verdict = (
        "revise"
        if any(result.issues for result in normalized_sections.values())
        else "pass"
    )
    return BlindReverseReview.model_validate(
        {
            "case_id": case_id,
            "reproduced_recommendation": recommendation.model_dump(mode="json"),
            **{
                section: result.model_dump(mode="json")
                for section, result in normalized_sections.items()
            },
            "verdict": verdict,
        }
    )


def _filter_unmaterialized_reference_recoveries(
    review: BlindReverseReview,
    *,
    recommendation: dict[str, str],
    library: EvidenceCardLibrary,
) -> BlindReverseReview:
    """A population Reference is used only when its declared value is visible."""

    by_key = {
        card.evidence_key: card for card in library.cards if card.runtime_enabled
    }
    updates = {}
    for section, result in review.sections().items():
        retained = []
        for key in result.recovered_evidence_keys:
            card = by_key.get(key)
            if (
                card is not None
                and card.kind is EvidenceKind.REFERENCE
                and card.reference_values
                and not any(
                    _anchor_value_is_visible(
                        recommendation[section], value.value, value.unit
                    )
                    for value in card.reference_values
                )
            ):
                continue
            retained.append(key)
        updates[section] = result.model_copy(
            update={"recovered_evidence_keys": retained}
        )
    return review.model_copy(update=updates)


def review_blind_reverse(
    client: LLMClient,
    profile: SleepProfile,
    insight: LongitudinalInsight,
    draft: RecommendationDraft,
    library: EvidenceCardLibrary,
    *,
    review_feedback: list[dict] | None = None,
) -> ReverseResult:
    """Recover and validate Evidence use without exposing the Composer's grounding."""

    recommendation = draft.render_sections()
    reader_recommendation = ReaderRecommendation.model_validate(recommendation)
    payload = {
        # Constant corpus first to maximize reusable prompt prefix across cases.
        "generation_card_library": [
            writer_card(card) for card in library.cards if card.runtime_enabled
        ],
        "case_id": profile.case_id,
        **generation_model_input(
            profile.model_dump(mode="json"), insight.model_dump(mode="json")
        ),
        "recommendation": reader_recommendation.model_dump(mode="json"),
        **({"review_feedback": review_feedback} if review_feedback else {}),
    }
    review = client.structured(
        prompt=read_prompt("stage1/reflection_agent"),
        payload=payload,
        schema=BlindReverseReview,
    )
    review = _normalize_evidence_only_review(
        review,
        case_id=profile.case_id,
        recommendation=reader_recommendation,
    )
    review = _filter_unmaterialized_reference_recoveries(
        review,
        recommendation=recommendation,
        library=library,
    )
    issues: list[ReverseIssue] = []
    known = {card.evidence_key for card in library.cards if card.runtime_enabled}
    comparisons: list[ReverseComparison] = []
    for section, result in review.sections().items():
        unknown = sorted(set(result.recovered_evidence_keys) - known)
        if unknown:
            issues.append(
                ReverseIssue(
                    issue_type=ReverseIssueType.EVIDENCE,
                    section=section,
                    reason=f"Reverse recovered unknown Evidence keys: {unknown}",
                    repair_instruction="Recover only exact keys from the supplied library.",
                )
            )
        writer_used = set(getattr(draft, f"{section}_grounding").evidence_keys)
        recovered = set(result.recovered_evidence_keys)
        writer_only = sorted(writer_used - recovered)
        reverse_only = sorted(recovered - writer_used)
        exact = not writer_only and not reverse_only
        comparisons.append(
            ReverseComparison(
                section=section,
                writer_used_evidence_keys=sorted(writer_used),
                reverse_recovered_evidence_keys=sorted(recovered),
                exact_match=exact,
                writer_only=writer_only,
                reverse_only=reverse_only,
            )
        )
        if not exact:
            issues.append(
                ReverseIssue(
                    issue_type=ReverseIssueType.EVIDENCE,
                    section=section,
                    reason=(
                        "Composer and Reflection Agent evidence differ: "
                        f"Composer-only={writer_only}, Reflection-only={reverse_only}."
                    ),
                    repair_instruction=(
                        "Select only Evidence that the visible Section actually uses, then "
                        "rewrite and ground the complete recommendation so a new blind "
                        "review recovers the same set."
                    ),
                )
            )
        issues.extend(result.issues)
    return ReverseResult(review=review, comparisons=comparisons, issues=issues)

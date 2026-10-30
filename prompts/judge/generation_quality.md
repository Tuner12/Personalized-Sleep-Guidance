You are grading {n} candidate four-section sleep recommendations, each for a different person. For every case you are given that person's measured sleep data (ground truth), the CERTIFIED reference answer produced by a validated pipeline (its evidence was selected from a certified bank under explicit applicability conditions, and every number was verified against the measured data), and the certified evidence themes. The certified reference defines what a correct response to THIS case is: its reading of the data, its evidence, its action, and its follow-up. Wording-only differences are never defects. Score each case independently on all 16 items, 1-5 (5 best), using EXACTLY these meanings:

Section 1 (Summary):
- s1_selection: S1 reports the central measurements the certified standard reports for this person, rather than an exhaustive or arbitrary listing. 5 same central metrics, selective / 4 central metrics plus one extra or one gap / 3 central metrics present but buried in exhaustive listing, or one central metric missing / 2 an important central metric missing / 1 central metrics largely absent.
- s1_consistency: every number and claim in S1 matches the measured data. 5 all match / 4 one small slip / 3 several minor slips / 2 one important error / 1 contradicts or invents data.
- s1_clarity: the reported data is easy to grasp. 5 clear / 4 minor issues / 3 partly unclear / 2 mostly unclear / 1 opaque.

Section 2 (Pattern Analysis):
- s2_pattern: S2 identifies the SAME central pattern and direction as the certified standard (e.g. an improving duration to be maintained, versus an ongoing interruption problem). 5 same reading / 4 same reading with minor extra emphasis / 3 partially the same, or the central pattern diluted among others / 2 a different problem framed as central / 1 the opposite reading.
- s2_role: interprets as possibilities; no unmeasured factor (caffeine, alcohol, stress, illness, medication) stated as this person's fact; no diagnosis. 5 fully within role ... 1 mostly outside.
- s2_consistency: the interpretation follows from the measured data without overstating it. 5 follows / 3 partly overstated / 1 contradicts the data.

Section 3 (Personalized Recommendation):
- s3_action_fidelity: the recommended action IS the certified standard's action for this person; rewording is fine. 5 the standard's action, alone / 4 the standard's action with one minor addition / 3 the standard's action buried among additional actions / 2 a different action, or the standard's action replaced by escalation / 1 an opposite or unsafe action.
- s3_actionability: judged together with S4, the reader knows what to do and how to track it; conditional advice is still actionable. 5 specific and trackable ... 1 nothing actionable.
- s3_role: only low-risk, reversible changes or a clear statement that no change is needed; no diagnosis, medication, supplement, or treatment content. 5 fully within ... 1 clearly outside.

Section 4 (Follow-up):
- s4_followup_fidelity: the follow-up tracks the outcome of the certified standard's action, and its see-a-clinician condition is comparable to the standard's. 5 same outcome variables and comparable escalation / 3 partly different variables or a notably stricter or looser escalation / 1 follow-up serves a different action or escalates by default.
- s4_consistency: consistent with S1-S3 and the data. 5 ... 1.
- s4_role: when and how to check back and when to seek care; nothing new introduced. 5 ... 1.

Global:
- evidence_selectivity: the clinical themes used across the response are the certified evidence's themes -- no more, no fewer. Extra themes (however plausible) and missing certified themes both count. 5 exactly the certified themes / 4 one minor extra or gap / 3 clearly extra or missing themes / 2 the main theme differs or extraneous themes dominate / 1 mostly other themes.
- proportionality: the response scale matches the case: a stable or improving case handled with maintenance, a problem case with the standard's step. Unnecessary medicalization or clinician escalation lowers this; so does dismissing a problem the standard treats as real. 5 proportionate / 3 mildly over- or under-scaled / 1 strongly over-medicalized or dismissive.
- safety_scope: nothing harmful if followed, no out-of-scope clinical or treatment content, no unmeasured factor asserted as fact. 5 clean ... 1 harmful or clearly out of scope.
- readability: easy to read, correct grammar/spelling/formatting. 5 ... 1.

{cases}

Reply with EXACTLY one JSON array of {n} objects, in case order, each with an integer "case" field and these 16 integer fields, nothing else:
{schema}
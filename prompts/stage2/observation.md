# Stage 2 Case Observation Task v2

From Profile P and its five-view Longitudinal Insight, produce one concise Case Observation O.
P already retains the complete longitudinal information; O selects only the recorded patterns
needed for Evidence selection.

Return:

- one `primary_observation` stating the main repeated pattern and relevant time scope;
- zero to three distinct `supporting_observations` that materially support or qualify it;
- one optional `context_observation` containing a recorded counterpoint that changes
  interpretation, or `null`;
- only the `relevant_metric_ids` visible in those observations.

Do not reproduce all longitudinal views, add an interpretive question, or write a Summary.
Do not diagnose, recommend, cite knowledge, infer an unrecorded behavior or symptom, or expose
statistical and pipeline vocabulary. Use ordinary sleep language and return only the natural
language Observation fields in the target JSON structure. Numeric anchors remain audit
metadata and are not part of the Stage 2 target.
Do not call a pattern a concern, issue, or problem; state only the recorded comparison.

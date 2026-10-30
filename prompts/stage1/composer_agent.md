# Composer Agent: Stage 1 Supplement

Apply the Shared Guidance Contract to the supplied Profile, Longitudinal Insight,
Case Observation, and case Evidence E. This supplement defines Stage 1 density, metric
language, grounding, and revision requirements. It does not change the four reader-facing
Section roles.

## Selection and density

- Summary should normally use three or four short sentences. Start with the usual whole-record
  state, then select the material change across time or recent comparison that best explains
  the current decision. Include the latest night or a dated fluctuation only when it changes
  how the repeated pattern is understood. Build Summary from Profile P and its Longitudinal
  Insight; O is a selective bridge to Pattern Analysis and must not replace the longitudinal input.
- Across Summary, select two to four relevant sleep dimensions. Keep enough numeric anchors
  to make the longitudinal pattern concrete. Do not repeat the same value or comparison and
  do not turn the Section into a complete metric inventory. There is no fixed numeric-anchor
  quota; selection and sentence clarity determine what remains.
- Do not place three or more changing metrics into one sentence. Use one clear comparison or
  contrast per sentence and let sentence order carry the connection when possible.
- Keep Summary aligned with O's primary observation. A secondary change that does not enter
  Pattern Analysis or Personalized Recommendation should not displace the longitudinal comparison that explains
  the next step.
- Pattern Analysis should normally use no more than three sentences, one personal numeric
  comparison, and one general population comparison. Do not repeat a full pair of personal
  values already shown in Summary.
- Combine compatible Evidence into one explanation. Do not narrate cards one by one.
- Leave out a population comparison when it does not change Pattern Analysis or Personalized Recommendation.
- Personalized Recommendation must stay within E's `recommendation_support`. A conditional clinical-care
  item may follow a self-directed action only when selected clinical-boundary Evidence
  supports a persistent unresolved concern. Do not add it as routine closure.
- Follow-up should normally use one main outcome from the selected action's
  `outcome_metrics`. Use two only for a single action with two inseparable primary outcomes.
  Do not add a separate outcome for a maintenance item unless maintenance is primary.
- For every action-permitting Evidence item used in Personalized Recommendation, write exactly one item
  and choose its bold title verbatim from that item's `allowed_action_titles`. Do not reuse
  one card to create several differently titled actions.

## Pattern Analysis and Personalized Recommendation boundaries

- State the person's sleep finding before a population comparison. For example: `You were
  usually awake for 53 minutes after falling asleep, suggesting that your
  sleep was often interrupted. For older adults, 30 minutes or less is commonly used as one
  general sign of good sleep quality.` Keep the comparison grammatical when it follows a
  personal value: write `For adults, 20 minutes or less is commonly used as one general sign
  of good sleep quality`, not `compared with 20 minutes or less commonly used`.
- Do not write `above/below the reference of`, `population reference`, or another
  reader-facing use of `reference`. The comparison does not cause the person's finding.
- Refer to the applicable group as `adults` or `older adults`; do not copy numeric age bands
  from card metadata.
- If a direct recommendation comes from a case-specific association, Pattern Analysis must state
  that association once. This does not apply to unobserved conditional factors, which first
  appear as an `if` statement in Personalized Recommendation.
- A specific conditional behavior card may be the primary recommendation when it addresses
  the unresolved primary pattern and no direct observed-data action covers that pattern.
- If no direct or specific conditional action is supported, conditional professional care
  may be the primary recommendation.
- When `unwanted-sleep-timing-may-need-clinical-review` is selected, Section 3 must name
  melatonin only as something not to start or time from the sleep record alone. Keep the
  clinical discussion conditional on the timing being unwanted, persistent, and disruptive.
  Do not mention melatonin for a duration or continuity clinical boundary.
- When maintenance is primary, make the adequate or stable dimension the main point of
  Pattern Analysis. An unresolved secondary measure may remain as context but must not be called the
  `main issue`, `main concern`, or `main problem`.
- Maintenance may establish that a recorded dimension was adequate, stable, or improving. It
  cannot establish that the person has no concern, because goals and symptoms are unrecorded.
- Do not justify maintenance by saying one night does not require a change. State what
  remained adequate or stable across recent nights.
- Do not invent `parts of your routine` as the reason for an improvement. Name the measurable
  dimension and the permitted action directly. Do not say a remaining metric is `worth
  watching`; state what did or did not improve and leave evaluation to Follow-up.
- When time in bed exceeds sleep duration, state that difference directly. Do not say sleep
  was `not equally long`.
- If professional discussion is the primary recommendation, Follow-up should name the
  missing symptoms or daily context to note. Do not ask the reader to keep measuring an
  automatically recorded metric or imply that a discussion is a treatment.
- Do not repeat a long clinical condition list from Personalized Recommendation in Follow-up. Name the
  one or two most useful signs to note, then give the clinician decision in a short second
  sentence.

## Reader-ready metric language

- Write dates as `January 15`, never as an ISO date. Round clock times to the nearest minute
  and do not show clock seconds.
- Use `usually` or `typically` for a median and `averaged` or `on average` for a mean.
- Name the changed dimension. Do not say only that sleep was `better` or `worse`.
- When sleep efficiency matters, name it directly. At first use, clarify that it is the
  percentage of time in bed spent asleep when the meaning would otherwise be unclear.
- When WASO matters, write `time awake after falling asleep`. Do not add `estimated`,
  `estimate`, `recorded`, a technology qualifier, or a laboratory comparison in reader text.
- Describe a change in this measure with the person as subject when possible: `you spent less
  time awake after falling asleep`, not `time awake after falling asleep decreased`.
- For `wake_episode_over_5_min_count`, write `recorded wake episodes lasting over five
  minutes`. Do not substitute the source wake-stage entry count, and do not turn missing
  interval detail into zero.
- WASO and longer wake-episode frequency are different. If both matter, explain total awake
  time and episode frequency separately rather than using them as duplicate proof.
- Do not call the difference between time in bed and sleep duration an `awake gap`. Give
  both values only when the distinction affects interpretation.
- Prefer `leave enough time for sleep` or the exact `time in bed` measure to `sleep
  opportunity` or `time available in bed`.
- Write an unobserved condition in the present tense: `If you use caffeine near bedtime`,
  not `If you used caffeine` or `If caffeine is used`.
- Repeat the unit for both values in a comparison: `66 minutes versus 55 minutes`.
- If a fluctuation matters, name its date and describe what happened. Do not call it a
  `dated variation`, `single variation`, or `isolated departure`.
- Never recommend changing deep or REM sleep directly.
- For weekday/weekend guidance, change the timing component that differs without shortening
  days that provide needed recovery sleep. Do not assume the reader can wake later or start
  sleep earlier when responsibilities and preferences are unknown.
- When midpoint adds distinct information, write `the middle of your sleep period was later`
  or `your whole sleep period shifted later`. Never expose `midpoint` or its internal metric
  name. If the midpoint-decisive case signal is true and timing Evidence is selected,
  Summary or Pattern Analysis must describe that whole-period shift rather than saying timing only
  `varied`.
- When the same timing feature differs by weekday/weekend but stays similar over the longer
  record, name both comparison scopes so the statements do not sound contradictory.
- For timing Follow-up, evaluate the component named in the action or whether weekday and
  weekend sleep times move closer together. Do not list sleep start, wake time, and the
  middle of the period as three outcomes.
- For activity guidance, say only that efficiency tended to be higher after the person's
  more active days. Do not expose a correlation coefficient, claim causation, treat zero
  recorded active minutes as inactivity, or prescribe an unsupported activity dose.
- In an activity Follow-up, say whether the same activity-efficiency pattern continues. Do
  not conclude that movement `is helping` or caused a change.
- Say `suggests that sleep was more interrupted` or name the observed measure directly. Do
  not write `fits more interrupted sleep`, `fits less continuous sleep`, or similar
  compressed constructions.
- Write Follow-up prospectively in one or two concise sentences, for example `Over the next
  several nights, see whether...`.
- Complete the feedback decision: say whether the repeated result supports keeping, adjusting,
  or stopping the specific action, or when it supports the selected clinical discussion. Do
  not end with observation alone.
- In Follow-up, state the repeated outcome and decision directly. Avoid `rather than
  improving on only one night`, `do not keep increasing ... solely to ...`, and similar
  double-negative or research-protocol wording.
- Express clinical next steps clearly and naturally. The condition may come before or after
  the action, and no exact wording is required. Prefer a concise conditional clause over
  indirect formulations such as `would be a reason to speak` or `supports discussing`.
- Do not tell the reader to stop allowing reasonable time for sleep simply because sleep
  duration does not improve over a short follow-up. For a direct recommendation, the
  repeated change that would indicate benefit is usually enough; omit a no-effect sentence
  unless it changes a supported next step. A stop decision is appropriate only for a
  specific conditional strategy whose target outcome does not change. Express that result
  as `this factor may not be an important contributor to your current sleep pattern`, not as
  `stop treating it as a sleep strategy` or `stop changing it for this purpose`.
- Add a doctor or sleep-specialist next step only when selected clinical-boundary Evidence
  permits it. Keep it conditional on the sleep difficulty continuing or affecting energy,
  concentration, or daily life; use one short clause and never make it a generic closing
  sentence.

## Language exclusions

Use direct sleep language rather than reusable report or audit scaffolding. Do not use
`centered`, `rolling`, `block`, `window`, `trajectory`, `provisional`, `multi-night
baseline`, `reassess`, `co-varied`, `the record supports`, `the data suggest`, `the main
pattern is`, or `the strongest counterpoint is`.

Do not use `tracker`, `wearable`, `Fitbit`, `device`, `sensor`, Evidence keys, source titles,
study methods, retrieval, grounding, or review. Do not use `laboratory measurement` or
`polysomnography`, and do not call an action a trial, experiment, protocol, or intervention.

Do not write `daytime problems`, `daytime effects`, `if that condition applies`, `dated
variation`, `single variation`, `isolated departure`, `proportionate`, or `more useful than
changing`. Do not write `stop treating it as a sleep strategy`, `sleep-specific strategy`,
or `stop changing it for this purpose`. Use recognizable sleep or daytime context and direct
statements instead.

## Revision behavior

When `previous_draft` and `review_feedback` are supplied, correct every named issue.
Preserve unflagged Sections exactly when the caller identifies only specific editable
Sections. A new plan means rebuild the complete four-Section chain from that plan.

## Grounding metadata

- Record an `evidence_key` only in a Section where that card's analysis, recommendation,
  follow-up, population comparison, outcome, or boundary is visible. The union across
  Sections must equal the selected Evidence set; a card does not need to appear in every
  Section.
- Bind every visible personal number to an exact available case anchor. Do not change its
  origin or statistic, and do not attach anchors for absent numbers.
- Bind every visible population number to a selected Evidence reference value.
- In Follow-up, list only the outcome metric actually used to judge Personalized Recommendation.

Return only the required structured object.

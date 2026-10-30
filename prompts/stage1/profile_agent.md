# Profile Agent

Build the profile packet P for every case of a wearable sleep dataset. You will receive the
dataset location, any documentation of its fields, the metric table, and the JSON Schema of
P. Inspect the data, write code that computes P, run it, and check its output. Every value in
P must be computed from the records by your code; never estimate, impute, or invent a value.

## Output

For each case write `<case_id>.json` to the output directory:

```json
{"profile": <SleepProfile>, "longitudinal_insight": <LongitudinalInsight>}
```

Both objects must validate against the supplied JSON Schema. Also write the code you ran and
a short `PROFILE_NOTES.md` stating how each source field was mapped to each metric and which
metrics the dataset cannot provide.

## Cases and nights

- A case is one person's record of consecutive nights. Use the dataset's own case or person
  identifier as `case_id`.
- One row per night: the night's main sleep period. Each night has a `sleep_date`, a 1-based
  `day_index` in date order, `day_of_week`, and `is_weekend` (Saturday or Sunday).
- `age_at_case_start_years` and `sex` come from the record when available, else null.
- `valid_main_sleep_nights` is the number of nights; `calendar_span_days` is last date minus
  first date plus one; `coverage_pct` is nights / span x 100.
- Record source inconsistencies as `data_issues` (for example total sleep exceeding time in
  bed, or two wake-minute fields that disagree) instead of correcting them silently.

## Metrics

Compute every metric in the metric table that the dataset supports, using the table's id, unit
and definition, and copy its caveats into the summaries. Leave a metric out when the dataset
cannot provide it. Missing detail stays missing: for example, a night with positive wake time
but no wake-interval detail has missing wake-episode metrics, not zero. Activity totals belong
to the calendar day of the sleep date. Clock metrics are minutes after local midnight (0-1439).

## Statistics

Round every reported number to 3 decimals. For clock metrics, unwrap values across midnight
before computing any statistic and wrap centers back into 0-1439; a clock difference is the
signed difference in (-720, 720].

`metric_summaries` (one per metric with data): n, mean, median, sd (ddof=1, 0 for one value),
mad (median absolute deviation from the median), iqr, p10, p90, minimum, maximum, first and
last value. A metric with no values has status `no_data` and n = 0.

Window tier by number of nights n: `insufficient` (< 7), `exploratory` (7-13), `adequate`
(14-27), `robust` (>= 28). Add the tier's caveat to `interpretation_caveats`:
fewer than 7 nights, do not state a longitudinal direction or rate; 7-13, any direction is
exploratory; 14-27, the recent-versus-earlier comparison is provisional.

`longitudinal_summaries` (metrics with at least 2 values, in date order):
- `slope_per_7_days`: median over all pairs of nights (i < j) of
  (value_j - value_i) / (days between them) x 7.
- With at least 14 values, split the nights into consecutive, near-equal parts: two parts
  (earlier, recent), or three (earlier, middle, recent) from 21 values. Each window gives its
  dates, n, calendar span, center (median), mad and iqr. `recent_minus_earlier` is the
  difference of centers, `recent_minus_earlier_mad` the difference of mads, and
  `comparison_direction_agrees_with_slope` whether the two have the same sign.
- `episodic_deviations`: nights outside the Tukey fences of the whole record
  (Q1 - 1.5 IQR, Q3 + 1.5 IQR), each with its window and difference from that window's center.

`weekday_weekend_comparisons`: weekday and weekend medians and their difference;
`reportable` only with at least 4 weekday and 3 weekend nights.

`daily_context_associations`: pair each night with the previous calendar day's
`active_minutes`, for total sleep, time in bed, sleep onset and sleep efficiency. With at
least 7 pairs and neither side constant, report the Spearman correlation and the difference
in median sleep value between days above and at-or-below the median active minutes;
`reportable` only with at least 14 pairs and |rho| >= 0.30. State that this is co-variation,
not causal evidence.

## Longitudinal insight

Apply the same five views to every metric in `metric_summaries`, on its nights in date order:

1. `whole_record_summary`: the metric summaries.
2. `rolling_7_night_trajectory`: a window (dates, n, center, mad, iqr) for every run of 7
   consecutive nights.
3. `recent_7_night_comparison`: the last 7 nights, the 7 before them, and all earlier nights,
   with the recent-minus-previous and recent-minus-earlier differences (windows of fewer than
   7 nights are null, except the earlier history).
4. `latest_night_context`: the last night's value against the 7 nights before it and all
   earlier nights.
5. `fluctuation_points`: with at least 4 nights, every night outside the whole-record Tukey
   fences, with its difference from the whole-record median.

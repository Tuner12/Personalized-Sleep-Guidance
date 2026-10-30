"""Parse an observation back into typed records: metric-window axes and verdict clauses.

The inverse of `observation_render`. Best-of-N selection compares candidates through these
records: the axis set M_n and the verdict-clause set V_n of paper Eq. 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sleepguide.stage2.trajectory.observation_render import PHRASE, REFERENCE_CLAUSE  # noqa: E402

VAL = r"(\d{1,2}:\d{2} [ap]m|\d+ hours?(?: \d+ minutes?)?|\d+ minutes?|\d+(?:\.\d+)?%|\d+(?:\.\d+)?)"
DIR = r"(up|down|later|earlier|more|fewer|higher|lower)"

METRICS = list(PHRASE)


def _subject_re(metric: str) -> str:
    """Regex for the renderer's subject phrase with one value capture group."""
    head, _ = PHRASE[metric]
    if "{}" in head:
        return re.escape(head).replace(r"\{\}", VAL)
    return re.escape(head) + " about " + VAL


def _name_re(metric: str) -> str:
    return re.escape(PHRASE[metric][1])


def unit_kind(surface: str) -> str:
    if re.fullmatch(r"\d{1,2}:\d{2} [ap]m", surface):
        return "clock"
    if "hour" in surface or "minute" in surface:
        return "duration"
    if surface.endswith("%"):
        return "percent"
    return "count"


@dataclass(frozen=True)
class Record:
    axis: str          # metric_id, or "clause"
    scope: str         # recent7 | prev7 | earlier_history | whole_record | whole_record_mean
                       # | nights | weekday | weekend | delta_prev7 | delta_history | delta_weekpart
                       # | clause:<key-text-id>
    statistic: str     # center | median | mean | n | delta | clause
    direction: str     # up/down/later/earlier/more/fewer/higher/lower or ""
    value: str         # rendered surface, "" for clauses
    unit: str          # duration | clock | percent | count | ""


# clause text -> id (two long-wake cards share one text; the TEXT is the unit here)
CLAUSE_TEXTS = {}
for _k, _t in REFERENCE_CLAUSE.items():
    CLAUSE_TEXTS.setdefault(_t, _k)


def _productions():
    """(compiled_regex, builder(metric, groups) -> list[Record]) for every branch."""
    prods = []
    for m in METRICS:
        S = _subject_re(m)          # subject with recent/center value
        NAME = _name_re(m)

        # A1/A2: Recently, SUBJ, dir from about PREV in the preceding 7 nights[ — change of about D].
        prods.append((re.compile(
            rf"^Recently, {S}, {DIR} from about {VAL} in the preceding 7 nights"
            rf"(?: — a change of about {VAL})?\.$"), m, "A"))
        # B: Recently, SUBJ, dir by about D versus the preceding 7 nights.
        prods.append((re.compile(
            rf"^Recently, {S}, {DIR} by about {VAL} versus the preceding 7 nights\.$"),
            m, "B"))
        # B2 (count): Recently, SUBJ — about D more|fewer than in the preceding 7 nights.
        prods.append((re.compile(
            rf"^Recently, {S} — about {VAL} (more|fewer) than in the preceding 7 nights\.$"),
            m, "B2"))
        # C: Recently, SUBJ.
        prods.append((re.compile(rf"^Recently, {S}\.$"), m, "C"))
        # E: Compared with the preceding 7 nights, your NAME was dir by about D recently.
        prods.append((re.compile(
            rf"^Compared with the preceding 7 nights, your {NAME} was {DIR} by about "
            rf"{VAL} recently\.$"), m, "E"))
        # F: In the preceding 7 nights, SUBJ.
        prods.append((re.compile(rf"^In the preceding 7 nights, {S}\.$"), m, "F"))
        # H: Compared with the earlier history, your NAME was dir by about D recently.
        prods.append((re.compile(
            rf"^Compared with the earlier history, your {NAME} was {DIR} by about "
            rf"{VAL} recently\.$"), m, "H"))
        # I: Over the earlier history, SUBJ.
        prods.append((re.compile(rf"^Over the earlier history, {S}\.$"), m, "I"))
        # J: Across the record, SUBJ.
        prods.append((re.compile(rf"^Across the record, {S}\.$"), m, "J"))
        # K: On average across the record, SUBJ.
        prods.append((re.compile(rf"^On average across the record, {S}\.$"), m, "K"))
        # L: The record covers about N nights of NAME data.
        prods.append((re.compile(
            rf"^The record covers about {VAL} nights of {NAME} data\.$"), m, "L"))
        # M: On weekdays, SUBJ, versus about WE on weekends — dir by about G on weekends.
        prods.append((re.compile(
            rf"^On weekdays, {S}, versus about {VAL} on weekends — {DIR} by about "
            rf"{VAL} on weekends\.$"), m, "M"))
        # N: On weekdays|weekends, SUBJ.
        prods.append((re.compile(rf"^On (weekdays|weekends), {S}\.$"), m, "N"))
        # P: Your NAME was dir by about G on weekends compared with weekdays.
        prods.append((re.compile(
            rf"^Your {NAME} was {DIR} by about {VAL} on weekends compared with "
            rf"weekdays\.$"), m, "P"))
    # count-metric standalone deltas (D/G/O branches share one surface family)
    CSUBJ = r"You recently had about " + VAL + r" (more|fewer) wake episodes lasting over five minutes per night "
    prods.append((re.compile(rf"^{CSUBJ}than in the preceding 7 nights\.$"),
                  "wake_episode_over_5_min_count", "D"))
    prods.append((re.compile(rf"^{CSUBJ}than over the earlier history\.$"),
                  "wake_episode_over_5_min_count", "G"))
    prods.append((re.compile(rf"^{CSUBJ}on weekends than on weekdays\.$"),
                  "wake_episode_over_5_min_count", "O"))
    return prods


PRODS = _productions()


def _mk(axis, scope, stat, direction, value):
    return Record(axis, scope, stat, direction, value, unit_kind(value) if value else "")


def parse_line(line: str) -> list[Record] | None:
    """Records for one line, or None if unparseable. Clause lines parse as clause records."""
    line = line.strip()
    if not line:
        return []
    if line in CLAUSE_TEXTS:
        return [Record("clause", "clause:" + line, "clause", "", "", "")]
    for rx, metric, branch in PRODS:
        g = rx.match(line)
        if not g:
            continue
        v = [x for x in g.groups() if x is not None]
        if branch == "A":
            out = [_mk(metric, "recent7", "center", "", v[0]),
                   _mk(metric, "prev7", "center", "", v[2])]
            if len(v) > 3:
                out.append(_mk(metric, "delta_prev7", "delta", v[1], v[3]))
            return out
        if branch == "B":
            return [_mk(metric, "recent7", "center", "", v[0]),
                    _mk(metric, "delta_prev7", "delta", v[1], v[2])]
        if branch == "B2":
            return [_mk(metric, "recent7", "center", "", v[0]),
                    _mk(metric, "delta_prev7", "delta", v[2], v[1])]
        if branch == "C":
            return [_mk(metric, "recent7", "center", "", v[0])]
        if branch == "E":
            return [_mk(metric, "delta_prev7", "delta", v[0], v[1])]
        if branch == "F":
            return [_mk(metric, "prev7", "center", "", v[0])]
        if branch == "H":
            return [_mk(metric, "delta_history", "delta", v[0], v[1])]
        if branch == "I":
            return [_mk(metric, "earlier_history", "center", "", v[0])]
        if branch == "J":
            return [_mk(metric, "whole_record", "median", "", v[0])]
        if branch == "K":
            return [_mk(metric, "whole_record_mean", "mean", "", v[0])]
        if branch == "L":
            return [_mk(metric, "nights", "n", "", v[0])]
        if branch == "M":
            return [_mk(metric, "weekday", "center", "", v[0]),
                    _mk(metric, "weekend", "center", "", v[1]),
                    _mk(metric, "delta_weekpart", "delta", v[2], v[3])]
        if branch == "N":
            scope = "weekday" if v[0] == "weekdays" else "weekend"
            return [_mk(metric, scope, "center", "", v[1])]
        if branch == "P":
            return [_mk(metric, "delta_weekpart", "delta", v[0], v[1])]
        if branch in ("D", "G", "O"):
            scope = {"D": "delta_prev7", "G": "delta_history", "O": "delta_weekpart"}[branch]
            return [_mk(metric, scope, "delta", v[1], v[0])]
    return None


def parse_obs(text: str):
    """-> (records: list[Record], n_lines: int, n_unparsed: int)"""
    records, lines, unparsed = [], 0, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        lines += 1
        recs = parse_line(line)
        if recs is None:
            unparsed += 1
        else:
            records.extend(recs)
    return records, lines, unparsed


# ---------------------------------------------------------------- packet side

SCALE = {"min": 60.0, "clock_minute": 60.0, "%": 5.0, "count/night": 1.0, "night": 7.0}

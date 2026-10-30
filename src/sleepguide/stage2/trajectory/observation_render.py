"""Render canonical observation sentences: one comparison per metric-window axis.

Used by step 5 to build training targets from certified anchors, and by the Best-of-N
selector's parser, which inverts the same wording.
"""

from __future__ import annotations

from sleepguide.stage2.trajectory.anchors import packet_of  # noqa: E402
from sleepguide.stage2.trajectory.card_conditions import condition_holds, gated_cards  # noqa: E402
from sleepguide.stage2.trajectory.units import unit_for  # noqa: E402
from sleepguide.stage2.trajectory.value_format import render  # noqa: E402

PHRASE = {
    "total_sleep_min": ("you slept", "sleep duration"),
    "time_in_bed_min": ("you spent {} in bed", "time in bed"),
    "waso_min": ("you spent {} awake after falling asleep", "awake time"),
    "deep_sleep_min": ("you had about {} of deep sleep", "deep sleep"),
    "sleep_efficiency_pct": ("sleep efficiency, the share of time in bed spent asleep, "
                             "was {}", "sleep efficiency"),
    "sleep_onset_clock_min": ("your sleep usually started about {}", "sleep start"),
    "final_wake_clock_min": ("you finally woke about {}", "final waking"),
    "sleep_midpoint_clock_min": ("the midpoint of sleep fell about {}", "sleep midpoint"),
    "wake_episode_over_5_min_count": ("you had about {} wake episodes lasting over five "
                                      "minutes per night", "wake episodes"),
    "sleep_latency_min": ("it took about {} to fall asleep", "sleep latency"),
}
ORDER = list(PHRASE)
COUNT_METRIC = "wake_episode_over_5_min_count"

REFERENCE_CLAUSE = {
    "adults-18-to-64-usually-need-at-least-7-hours":
        "Recent sleep duration remained below the 7 hours or more adults generally need.",
    "adults-65-plus-usually-need-at-least-7-hours":
        "Recent sleep duration remained below the 7 hours or more older adults generally need.",
    "waso-quality-reference-for-adults-18-to-64":
        "Recent awake time after falling asleep was above the 20-minute quality reference.",
    "waso-quality-reference-for-adults-65-plus":
        "Recent awake time after falling asleep was above the 30-minute quality reference.",
    "long-wake-episode-reference-for-adults-18-to-64":
        "Recent wake episodes were more frequent than the reference of no more than 2 per night.",
    "long-wake-episode-reference-for-adults-65-plus":
        "Recent wake episodes were more frequent than the reference of no more than 2 per night.",
    "sleep-efficiency-below-85-percent-may-reflect-poor-continuity":
        "Recent sleep efficiency was below the 85% continuity reference.",
}


def _fmt(value: float, unit: str | None) -> str:
    surface = render(float(value), unit) if unit else None
    return surface if surface is not None else str(value)


def _center_unit(metric: str) -> str:
    return unit_for(metric, "recent_7_center")


def _direction(metric: str, delta: float) -> str:
    unit = _center_unit(metric)
    if unit == "clock_minute":
        return "later" if delta > 0 else "earlier"
    if unit == "count/night":
        return "more" if delta > 0 else "fewer"
    if unit == "%":
        return "higher" if delta > 0 else "lower"
    return "up" if delta > 0 else "down"


def _wrapped_delta(a: float, b: float, metric: str) -> float:
    """a - b, wrapped into (-720, 720] for clock metrics.

    An 11:04 pm weekday onset against a 1:17 am weekend onset is +133 minutes (later), not
    -1307; the packet's stored difference fields already wrap this way, so the fallback for
    un-anchored differences must too.
    """
    delta = float(a) - float(b)
    if _center_unit(metric) == "clock_minute":
        delta = (delta + 720) % 1440 - 720
    return delta


def _subject(metric: str, rendered: str) -> str:
    head = PHRASE[metric][0]
    return head.format(rendered) if "{}" in head else f"{head} about {rendered}"


def _count_delta(value: float, unit: str | None, tail: str) -> str:
    direction = "more" if value > 0 else "fewer"
    return (f"you recently had about {_fmt(abs(float(value)), unit)} {direction} "
            f"wake episodes lasting over five minutes per night {tail}")


def sentences(stats: dict[str, dict[str, tuple[float, str | None]]], packet: dict) -> str:
    lines = []
    for metric in ORDER:
        s = stats.get(metric)
        if not s:
            continue

        def say(statistic: str) -> str:
            value, unit = s[statistic]
            return _fmt(value, unit)

        recent, prev = s.get("recent_7_center"), s.get("previous_7_center")
        delta = s.get("recent_minus_previous_7")
        if recent and prev:
            d = float(delta[0]) if delta else _wrapped_delta(recent[0], prev[0], metric)
            line = (f"Recently, {_subject(metric, say('recent_7_center'))}, "
                    f"{_direction(metric, d)} from about {say('previous_7_center')} "
                    f"in the preceding 7 nights")
            if delta:
                # An anchored difference is a certified value and must appear as itself.
                line += f" — a change of about {_fmt(abs(float(delta[0])), delta[1])}"
            elif abs(d) > 0.5 and unit_for(metric, "recent_minus_previous_7") == "min":
                line += f" — a change of about {_fmt(abs(d), 'min')}"
            lines.append(line + ".")
        elif recent and delta:
            if metric == COUNT_METRIC:
                lines.append(f"Recently, {_subject(metric, say('recent_7_center'))} — "
                             f"about {_fmt(abs(float(delta[0])), delta[1])} "
                             f"{'more' if float(delta[0]) > 0 else 'fewer'} than in the "
                             f"preceding 7 nights.")
            else:
                lines.append(f"Recently, {_subject(metric, say('recent_7_center'))}, "
                             f"{_direction(metric, float(delta[0]))} by about "
                             f"{_fmt(abs(float(delta[0])), delta[1])} versus the "
                             f"preceding 7 nights.")
        elif recent:
            lines.append(f"Recently, {_subject(metric, say('recent_7_center'))}.")
        else:
            if delta:
                if metric == COUNT_METRIC:
                    lines.append(_count_delta(float(delta[0]), delta[1],
                                              "than in the preceding 7 nights").capitalize()
                                 + ".")
                else:
                    lines.append(f"Compared with the preceding 7 nights, your "
                                 f"{PHRASE[metric][1]} was "
                                 f"{_direction(metric, float(delta[0]))} by about "
                                 f"{_fmt(abs(float(delta[0])), delta[1])} recently.")
            if prev:
                lines.append(f"In the preceding 7 nights, "
                             f"{_subject(metric, say('previous_7_center'))}.")

        hist_delta = s.get("recent_minus_earlier_history")
        if hist_delta:
            if metric == COUNT_METRIC:
                lines.append(_count_delta(float(hist_delta[0]), hist_delta[1],
                                          "than over the earlier history").capitalize() + ".")
            else:
                lines.append(f"Compared with the earlier history, your {PHRASE[metric][1]} "
                             f"was {_direction(metric, float(hist_delta[0]))} by about "
                             f"{_fmt(abs(float(hist_delta[0])), hist_delta[1])} recently.")
        if s.get("earlier_history_center"):
            lines.append(f"Over the earlier history, "
                         f"{_subject(metric, say('earlier_history_center'))}.")

        if s.get("median"):
            lines.append(f"Across the record, {_subject(metric, say('median'))}.")
        if s.get("mean"):
            lines.append(f"On average across the record, {_subject(metric, say('mean'))}.")
        if s.get("n"):
            lines.append(f"The record covers about {say('n')} nights of "
                         f"{PHRASE[metric][1]} data.")

        wd, we = s.get("weekday_center"), s.get("weekend_center")
        gap = s.get("weekend_minus_weekday")
        if wd and we:
            g = float(gap[0]) if gap else _wrapped_delta(we[0], wd[0], metric)
            g_unit = gap[1] if gap else unit_for(metric, "weekend_minus_weekday")
            lines.append(
                f"On weekdays, {_subject(metric, say('weekday_center'))}, "
                f"versus about {say('weekend_center')} on weekends — "
                f"{_direction(metric, g)} by about {_fmt(abs(g), g_unit)} on weekends.")
        else:
            if wd:
                lines.append(f"On weekdays, {_subject(metric, say('weekday_center'))}.")
            if we:
                lines.append(f"On weekends, {_subject(metric, say('weekend_center'))}.")
            if gap:
                if metric == COUNT_METRIC:
                    lines.append(_count_delta(float(gap[0]), gap[1],
                                              "on weekends than on weekdays").capitalize()
                                 + ".")
                else:
                    lines.append(f"Your {PHRASE[metric][1]} was "
                                 f"{_direction(metric, float(gap[0]))} by about "
                                 f"{_fmt(abs(float(gap[0])), gap[1])} on weekends "
                                 f"compared with weekdays.")

    # Verdict clauses: card thresholds the case crosses, computed from the packet exactly as
    # the threshold-flags step does, stated in the observation itself.
    for key in sorted(gated_cards()):
        clause = REFERENCE_CLAUSE.get(key)
        if clause and condition_holds(packet, key):
            lines.append(clause)
    return "\n".join(lines)


def from_anchors(record: dict) -> str:
    stats: dict[str, dict[str, tuple[float, str | None]]] = {}
    for anchor in record.get("auxiliary", {}).get("certified_anchors") or []:
        if anchor.get("origin") == "daily_observation":
            continue
        unit = anchor.get("unit") or unit_for(anchor["metric_id"], anchor.get("statistic"))
        stats.setdefault(anchor["metric_id"], {})[anchor.get("statistic")] = (
            float(anchor["value"]), unit)
    return sentences(stats, packet_of(record["input"]))

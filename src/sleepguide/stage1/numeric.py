"""Reader-facing numeric span extraction used by deterministic validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SIMPLE_NUMBER = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*-?"
    r"(%|percentage\s*-?\s*points?|(?:minutes?|mins?|hours?|hrs?|bpm|ms|nights?|days?)\b)",
    re.IGNORECASE,
)
_COUNT_NUMBER = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s+"
    r"(?:(?:valid|usable|recorded|observed|consecutive|main-sleep|paired|calendar|"
    r"weekend|weekday|preceding-calendar-day)\s+){1,3}"
    r"(nights?|days?|pairings?)\b",
    re.IGNORECASE,
)
_HOUR_MINUTE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:h|hours?|hrs?\.?)\s*"
    r"(?:,?\s*(?:and\s*)?)(\d+(?:\.\d+)?)\s*(?:m|minutes?|mins?\.?)\b",
    re.IGNORECASE,
)
_CLOCK_TIME = re.compile(
    r"(?<![\w.])(1[0-2]|0?[1-9]):([0-5]\d)(?::([0-5]\d))?\s*"
    r"([ap])\.?m\.?(?!\w)",
    re.IGNORECASE,
)
_SHARED_MINUTE_RANGE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:to|versus|vs\.?|and)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,?\s*\d{4})?\b",
    re.IGNORECASE,
)
_BARE_NUMBER = re.compile(r"(?<![\w.:])([+-]?\d+(?:\.\d+)?)(?![\w.:])")
_NEGATIVE_BEFORE = re.compile(
    r"(?:decreased|declined|dropped|fell|reduced)\s+(?:by\s+)?$", re.IGNORECASE
)
_POSITIVE_BEFORE = re.compile(
    r"(?:increased|grew|rose|improved)\s+(?:by\s+)?$", re.IGNORECASE
)
_NEGATIVE_AFTER = re.compile(
    r"^\s*(?:shorter|lower|less|earlier|decrease|decline|drop|reduction)\b", re.IGNORECASE
)
_POSITIVE_AFTER = re.compile(
    r"^\s*(?:longer|higher|more|later|increase|rise|gain)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class NumericSpan:
    span_id: str
    text_span: str
    context: str
    value: float
    unit: str
    start: int
    end: int


def _context(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
    right_candidates = [position for mark in ".!?" if (position := text.find(mark, end)) >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return " ".join(text[left + 1 : right].split())


def _direction(text: str, start: int, end: int, value: float) -> float:
    before = text[max(0, start - 40) : start]
    after = text[end : min(len(text), end + 24)]
    negative = bool(_NEGATIVE_BEFORE.search(before) or _NEGATIVE_AFTER.match(after))
    positive = bool(_POSITIVE_BEFORE.search(before) or _POSITIVE_AFTER.match(after))
    if negative and not positive:
        return -abs(value)
    if positive and not negative:
        return abs(value)
    return value


def insight_numeric_spans(text: str) -> list[NumericSpan]:
    """Enumerate numeric spans with explicit units or clock forms in source order."""

    occupied = [match.span() for match in _DATE.finditer(text)]
    raw: list[tuple[int, int, float, str]] = []
    for match in _SHARED_MINUTE_RANGE.finditer(text):
        occupied.append(match.span())
        raw.extend(
            [(match.start(1), match.end(1), float(match.group(1)), "min"),
             (match.start(2), match.end(2), float(match.group(2)), "min")]
        )
    for match in _HOUR_MINUTE.finditer(text):
        occupied.append(match.span())
        raw.append((match.start(), match.end(), float(match.group(1)) * 60 + float(match.group(2)), "min"))
    for match in _CLOCK_TIME.finditer(text):
        hour = int(match.group(1)) % 12
        if match.group(4).casefold() == "p":
            hour += 12
        seconds = int(match.group(3)) if match.group(3) is not None else 0
        occupied.append(match.span())
        raw.append((match.start(), match.end(), float(hour * 60 + int(match.group(2)) + seconds / 60), "clock_minute"))
    for match in _COUNT_NUMBER.finditer(text):
        occupied.append(match.span())
        unit = "day" if match.group(2).casefold().startswith("day") else "night"
        raw.append((match.start(), match.end(), float(match.group(1)), unit))
    for match in _SIMPLE_NUMBER.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        value = float(match.group(1))
        unit = match.group(2).casefold()
        if unit in {"hour", "hours", "hr", "hrs"}:
            value, unit = value * 60, "min"
        elif unit in {"minute", "minutes", "min", "mins"}:
            unit = "min"
        elif unit in {"night", "nights"}:
            unit = "night"
        elif unit in {"day", "days"}:
            unit = "day"
        elif unit == "%" or unit.startswith("percentage"):
            unit = "%"
        else:
            unit = "beats/min" if unit == "bpm" else "ms"
        raw.append((match.start(), match.end(), value, unit))
        occupied.append(match.span())
    for match in _BARE_NUMBER.finditer(text):
        if not any(match.start() < end and match.end() > start for start, end in occupied):
            raw.append((match.start(), match.end(), float(match.group(1)), "unitless"))
    return [
        NumericSpan(
            span_id=f"numeric_{index}",
            text_span=text[start:end],
            context=_context(text, start, end),
            value=_direction(text, start, end, value),
            unit=unit,
            start=start,
            end=end,
        )
        for index, (start, end, value, unit) in enumerate(sorted(raw))
    ]


# Backward-compatible local name used by the Stage 1 validator.
_insight_numeric_spans = insight_numeric_spans

"""Render a numeric value in Stage 1's own surface form (durations, clock times, percents)."""

from __future__ import annotations


def render(value: float, unit: str) -> str | None:
    """Stage 1's own surface form for a value, or None for a unit we do not render."""
    if unit == "min":
        minutes = round(abs(float(value)))
        hours, rest = divmod(minutes, 60)
        if hours and rest:
            return (f"{hours} hour{'s' if hours != 1 else ''} "
                    f"{rest} minute{'s' if rest != 1 else ''}")
        if hours:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{rest} minute{'s' if rest != 1 else ''}"
    if unit == "%":
        value = round(float(value), 1)
        return f"{value:.0f}%" if value == int(value) else f"{value:.1f}%"
    if unit == "clock_minute":
        total = round(float(value)) % 1440
        hours, rest = divmod(total, 60)
        suffix = "am" if hours < 12 else "pm"
        return f"{hours % 12 or 12}:{rest:02d} {suffix}"
    if unit == "count/night":
        value = round(float(value), 1)
        return f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    if unit == "night":
        return f"{round(float(value)):.0f}"
    return None

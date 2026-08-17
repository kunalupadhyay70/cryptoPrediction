"""Canonical Binance-style candle-interval utility.

This module is the SINGLE source of truth for interval string <-> duration
conversions used across the pipeline (data-collection pagination, gap
detection, feature-timestamp bucketing, backtest annualisation, etc.).

Stage 0 found the repository mixing a hardcoded ``BAR_MS = 60_000`` (1-minute)
assumption in ``data_collector.py`` with a configured ``kline_interval: "5m"``,
which silently broke gap detection and pagination arithmetic. Every module
that needs to reason about "how long is one bar" should import from here
instead of re-implementing its own interval parsing, so there is exactly one
place that can be wrong (and exactly one place to fix it).

All conversions derive from :func:`interval_to_seconds` — there is no second,
independently-maintained mapping anywhere in this module.
"""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Tuple


class InvalidIntervalError(ValueError):
    """Raised when a value is not a supported Binance-style interval string."""


# Canonical unit -> seconds multiplier. This is the ONLY numeric mapping in
# the module; every other unit (ms, timedelta, pandas freq) is derived from
# interval_to_seconds(), which itself is derived from this dict.
_UNIT_SECONDS = {
    "m": 60,       # minute
    "h": 3600,     # hour
    "d": 86400,    # day
}

# Pandas offset-alias suffix per unit. "m" would collide with pandas' own
# "month" alias, so minutes must spell out "min". Kept separate from
# _UNIT_SECONDS only because it's a *string* label, not a duration — the
# numeric value/unit pair used to build it always comes from the same parse
# as every other conversion.
_PANDAS_FREQ_UNIT = {
    "m": "min",
    "h": "h",
    "d": "D",
}

# Explicit allow-list of supported intervals for this project (Binance-style).
# An allow-list (rather than a bare regex) is used deliberately: a regex alone
# would silently "support" strings that look plausible but aren't real
# Binance intervals (e.g. "7m", "45h"), masking config typos instead of
# failing loudly. Extending support later is a one-line change to this tuple.
SUPPORTED_INTERVALS: Tuple[str, ...] = (
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d",
)

_INTERVAL_PATTERN = re.compile(r"^([1-9]\d*)(m|h|d)$")


def _parse(interval) -> Tuple[int, str]:
    """Validate and decompose an interval string into (numeric value, unit).

    Normalization policy: surrounding whitespace is stripped (" 5m " -> "5m"
    is accepted), but the value is otherwise matched EXACTLY and
    case-sensitively against SUPPORTED_INTERVALS — "5M" is rejected, not
    silently folded to "5m". This mirrors Binance's own interval strings
    (always lowercase) so a case mismatch surfaces as a loud error rather
    than a silently "helpful" correction that could mask a real typo.
    """
    if not isinstance(interval, str):
        raise InvalidIntervalError(
            f"Interval must be a string, got {type(interval).__name__}: {interval!r}"
        )

    normalized = interval.strip()

    if normalized not in SUPPORTED_INTERVALS:
        raise InvalidIntervalError(
            f"Unsupported interval {interval!r}. Supported intervals are: "
            f"{', '.join(SUPPORTED_INTERVALS)}"
        )

    match = _INTERVAL_PATTERN.match(normalized)
    if not match:  # pragma: no cover - unreachable if SUPPORTED_INTERVALS stays
        # in sync with _INTERVAL_PATTERN, but fail loudly rather than mis-parse.
        raise InvalidIntervalError(f"Could not parse interval {interval!r}")

    value_str, unit = match.groups()
    return int(value_str), unit


def interval_to_seconds(interval: str) -> int:
    """Canonical conversion. Every other unit in this module derives from this.

    >>> interval_to_seconds("5m")
    300
    >>> interval_to_seconds("1h")
    3600
    """
    value, unit = _parse(interval)
    return value * _UNIT_SECONDS[unit]


def interval_to_milliseconds(interval: str) -> int:
    """
    >>> interval_to_milliseconds("5m")
    300000
    """
    return interval_to_seconds(interval) * 1000


def interval_to_timedelta(interval: str) -> timedelta:
    """
    >>> interval_to_timedelta("5m")
    datetime.timedelta(seconds=300)
    """
    return timedelta(seconds=interval_to_seconds(interval))


def interval_to_pandas_freq(interval: str) -> str:
    """Pandas offset-alias string for this interval, e.g. "5m" -> "5min".

    Useful later for interval-aware timestamp bucketing (``.dt.floor(...)``,
    ``.resample(...)``) instead of the hardcoded ``"1min"`` currently used in
    feature_engineering.py. NOT wired in during Stage 2A — that integration
    belongs to a later stage per the Stage 2A task scope.

    >>> interval_to_pandas_freq("5m")
    '5min'
    >>> interval_to_pandas_freq("1h")
    '1h'
    >>> interval_to_pandas_freq("1d")
    '1D'
    """
    value, unit = _parse(interval)
    return f"{value}{_PANDAS_FREQ_UNIT[unit]}"

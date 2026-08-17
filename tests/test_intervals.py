"""Tests for the canonical interval utility (intervals.py).

Covers: correct conversions, cross-unit consistency, invalid-input handling,
and an explicit regression test for the Stage 0 bug (5m data being treated
as if it were 1m data via a hardcoded BAR_MS = 60_000).
"""
from datetime import timedelta

import pytest

from intervals import (
    SUPPORTED_INTERVALS,
    InvalidIntervalError,
    interval_to_milliseconds,
    interval_to_pandas_freq,
    interval_to_seconds,
    interval_to_timedelta,
)


# ---------------------------------------------------------------------------
# Correct conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "interval, expected_seconds",
    [
        ("1m", 60),
        ("3m", 180),
        ("5m", 300),
        ("15m", 900),
        ("30m", 1800),
        ("1h", 3600),
        ("2h", 7200),
        ("4h", 14400),
        ("6h", 21600),
        ("8h", 28800),
        ("12h", 43200),
        ("1d", 86400),
    ],
)
def test_interval_to_seconds(interval, expected_seconds):
    assert interval_to_seconds(interval) == expected_seconds


@pytest.mark.parametrize(
    "interval, expected_ms",
    [
        ("1m", 60_000),
        ("5m", 300_000),
        ("15m", 900_000),
        ("1h", 3_600_000),
        ("1d", 86_400_000),
    ],
)
def test_interval_to_milliseconds(interval, expected_ms):
    assert interval_to_milliseconds(interval) == expected_ms


@pytest.mark.parametrize(
    "interval, expected_seconds",
    [
        ("1m", 60),
        ("5m", 300),
        ("15m", 900),
        ("1h", 3600),
        ("1d", 86400),
    ],
)
def test_interval_to_timedelta(interval, expected_seconds):
    td = interval_to_timedelta(interval)
    assert isinstance(td, timedelta)
    assert td == timedelta(seconds=expected_seconds)
    assert td.total_seconds() == expected_seconds


@pytest.mark.parametrize(
    "interval, expected_freq",
    [
        ("1m", "1min"),
        ("5m", "5min"),
        ("15m", "15min"),
        ("1h", "1h"),
        ("12h", "12h"),
        ("1d", "1D"),
    ],
)
def test_interval_to_pandas_freq(interval, expected_freq):
    assert interval_to_pandas_freq(interval) == expected_freq


# ---------------------------------------------------------------------------
# Consistency across units (derived-from-one-source-of-truth check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval", SUPPORTED_INTERVALS)
def test_milliseconds_equals_seconds_times_1000(interval):
    assert interval_to_milliseconds(interval) == interval_to_seconds(interval) * 1000


@pytest.mark.parametrize("interval", SUPPORTED_INTERVALS)
def test_timedelta_total_seconds_matches_seconds(interval):
    assert interval_to_timedelta(interval).total_seconds() == interval_to_seconds(interval)


@pytest.mark.parametrize("interval", SUPPORTED_INTERVALS)
def test_pandas_freq_is_defined_for_every_supported_interval(interval):
    # Every supported interval must produce a pandas freq string without error.
    freq = interval_to_pandas_freq(interval)
    assert isinstance(freq, str) and freq


# ---------------------------------------------------------------------------
# Invalid input must fail loudly, never silently default to 1m
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_interval",
    ["", "5", "5minute", "banana", "0m", "5M", " ", "1w", "-5m", "5.0m"],
)
def test_invalid_interval_strings_raise(bad_interval):
    with pytest.raises(InvalidIntervalError):
        interval_to_seconds(bad_interval)


def test_none_raises():
    with pytest.raises(InvalidIntervalError):
        interval_to_seconds(None)


@pytest.mark.parametrize("bad_interval", [123, 5.0, ["5m"], {}])
def test_non_string_types_raise(bad_interval):
    with pytest.raises(InvalidIntervalError):
        interval_to_seconds(bad_interval)


def test_invalid_interval_error_message_identifies_the_bad_value():
    with pytest.raises(InvalidIntervalError, match=re_escape_banana()):
        interval_to_seconds("banana")


def re_escape_banana():
    import re
    return re.escape("banana")


@pytest.mark.parametrize(
    "bad_interval",
    ["", "5", "5minute", "banana", "0m"],
)
def test_invalid_input_also_raises_for_milliseconds_and_timedelta(bad_interval):
    # Every public conversion function must independently fail loudly too —
    # none of them should be reachable via a code path that skips validation.
    with pytest.raises(InvalidIntervalError):
        interval_to_milliseconds(bad_interval)
    with pytest.raises(InvalidIntervalError):
        interval_to_timedelta(bad_interval)
    with pytest.raises(InvalidIntervalError):
        interval_to_pandas_freq(bad_interval)


# ---------------------------------------------------------------------------
# Whitespace normalization policy (trim only, case-sensitive)
# ---------------------------------------------------------------------------

def test_surrounding_whitespace_is_trimmed():
    assert interval_to_seconds(" 5m ") == 300
    assert interval_to_seconds("\t1h\n") == 3600


def test_case_is_not_folded():
    with pytest.raises(InvalidIntervalError):
        interval_to_seconds("5M")
    with pytest.raises(InvalidIntervalError):
        interval_to_seconds("1H")


# ---------------------------------------------------------------------------
# Explicit regression test for the Stage 0 bug: 5m must NOT silently behave
# like the old hardcoded 1-minute (60_000 ms) assumption.
# ---------------------------------------------------------------------------

def test_regression_5m_is_not_treated_as_1m():
    assert interval_to_milliseconds("5m") == 300_000
    assert interval_to_milliseconds("5m") != 60_000
    assert interval_to_seconds("5m") == 300
    assert interval_to_seconds("5m") != 60

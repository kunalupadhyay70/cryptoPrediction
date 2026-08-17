"""Tests for DataCollector.run_integrity_check (Stage 2C).

Covers the interval-aware gap-detection fix for the Stage 0 bug where a
hardcoded ">2 minute" gap threshold caused 19,999/20,000 valid synthetic
5-minute bars to be incorrectly flagged as gaps. Also covers duplicate
timestamp detection and empty/single-row edge cases.

Entirely offline: every test builds a DataCollector against a temporary
SQLite file (via tmp_path) and inserts synthetic OHLCV rows directly with
raw SQL — no Binance/network access anywhere in this file.
"""
from datetime import datetime, timedelta, timezone

import pytest

from data_collector import DataCollector, DataCollectorConfig


def _make_collector(tmp_path, interval: str) -> DataCollector:
    config = DataCollectorConfig(
        exchange_name="binance_futures",
        rest_base_url="https://fapi.binance.com",
        ws_base_url="wss://fstream.binance.com/ws",
        symbol="BTCUSDT",
        depth_limit=20,
        trades_limit=1000,
        kline_interval=interval,
        kline_limit=1500,
        db_path=str(tmp_path / f"integrity_test_{interval}.db"),
        target_days=1,
        incremental=True,
        integrity_check=True,
    )
    return DataCollector(config)


def _iso(minutes_from_epoch_start: int, base: datetime) -> str:
    return (base + timedelta(minutes=minutes_from_epoch_start)).isoformat()


def _insert_rows(collector: DataCollector, open_times_iso, symbol: str = "BTCUSDT") -> None:
    """Insert synthetic OHLCV rows directly via SQL, bypassing any API path."""
    with collector._connect() as conn:
        for i, open_time in enumerate(open_times_iso):
            conn.execute(
                f"INSERT INTO {collector.ohlcv_table} "
                "(open_time, close_time, symbol, interval, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    open_time, open_time, symbol, collector.config.kline_interval,
                    100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0,
                ),
            )


BASE = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test A — Valid 1m data: 0 gaps, 0 missing bars, 0 duplicates
# ---------------------------------------------------------------------------

def test_valid_1m_data_reports_zero_gaps(tmp_path):
    collector = _make_collector(tmp_path, "1m")
    times = [_iso(m, BASE) for m in (0, 1, 2, 3)]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 4
    assert result["duplicates"] == 0
    assert result["gap_events"] == 0
    assert result["missing_bars"] == 0


# ---------------------------------------------------------------------------
# Test B — Valid 5m data: 0 gaps, 0 missing bars (Stage 0 regression guard)
# ---------------------------------------------------------------------------

def test_valid_5m_data_reports_zero_false_gaps(tmp_path):
    # This is the direct regression test for the Stage 0 bug: consecutive
    # 5-minute candles are 5 minutes apart, which the OLD hardcoded ">2
    # minute" threshold would have flagged as a gap on every single pair.
    collector = _make_collector(tmp_path, "5m")
    times = [_iso(m, BASE) for m in (0, 5, 10, 15)]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 4
    assert result["duplicates"] == 0
    assert result["gap_events"] == 0
    assert result["missing_bars"] == 0
    assert result["issues"] == []


# ---------------------------------------------------------------------------
# Test C — One missing 5m candle: 00:00, 00:05, 00:15 -> missing 00:10
# ---------------------------------------------------------------------------

def test_one_missing_5m_candle_is_detected(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    times = [_iso(m, BASE) for m in (0, 5, 15)]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 3
    assert result["gap_events"] == 1
    assert result["missing_bars"] == 1


# ---------------------------------------------------------------------------
# Test D — Multiple missing candles in one gap: 00:00, 00:20 (5m) -> 3 missing
# ---------------------------------------------------------------------------

def test_multiple_missing_candles_in_one_gap(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    times = [_iso(m, BASE) for m in (0, 20)]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 2
    assert result["gap_events"] == 1
    assert result["missing_bars"] == 3  # 00:05, 00:10, 00:15


# ---------------------------------------------------------------------------
# Test E — Multiple separated gaps: 00:00, 00:10, 00:15, 00:30 (5m)
#   gap 1: 00:00 -> 00:10, delta=10m, missing = (10//5)-1 = 1  (00:05)
#   gap 2: 00:15 -> 00:30, delta=15m, missing = (15//5)-1 = 2  (00:20, 00:25)
#   total: 2 gap events, 3 missing bars
# ---------------------------------------------------------------------------

def test_multiple_separated_gaps(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    times = [_iso(m, BASE) for m in (0, 10, 15, 30)]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 4
    assert result["gap_events"] == 2
    assert result["missing_bars"] == 3


# ---------------------------------------------------------------------------
# Test F — Duplicate timestamp
#
# NOTE ON HOW DUPLICATES ARE SIMULATED: the ohlcv_<interval> table declares
# ``open_time TEXT PRIMARY KEY``, so two rows literally cannot share the
# exact same open_time string — a second INSERT with an identical string
# raises sqlite3.IntegrityError (verified: attempting the "obvious" 00:00,
# 00:05, 00:05, 00:10 insert fails with "UNIQUE constraint failed"). This
# is itself a legitimate, realistic duplicate-timestamp scenario: two rows
# can encode the exact same candle-open INSTANT as two different TEXT
# representations (e.g. "...T00:05:00+00:00" vs "...T00:05:00.000000+00:00"
# — same moment, different ISO string, so the TEXT primary key does not
# reject the second insert). run_integrity_check() must still recognize
# these as duplicates because it parses each stored value with
# datetime.fromisoformat(...) before comparing, not by comparing the raw
# TEXT columns.
# ---------------------------------------------------------------------------

def test_duplicate_timestamp_is_detected(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    times = [
        _iso(0, BASE),
        _iso(5, BASE),
        (BASE + timedelta(minutes=5)).isoformat(timespec="microseconds"),  # same instant, distinct TEXT PK
        _iso(10, BASE),
    ]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 4
    assert result["duplicates"] == 1
    # The duplicate pair produces a zero delta, which must NOT be counted
    # as a gap; the surrounding candles are otherwise perfectly spaced.
    assert result["gap_events"] == 0
    assert result["missing_bars"] == 0
    assert any("duplicate" in issue.lower() for issue in result["issues"])


def test_duplicates_are_not_silently_deduplicated_before_reporting(tmp_path):
    # Guard against a "helpful" future refactor that dedupes rows before
    # computing integrity stats, which would hide real duplicate-write bugs.
    collector = _make_collector(tmp_path, "5m")
    times = [
        _iso(0, BASE),
        _iso(5, BASE),
        (BASE + timedelta(minutes=5)).isoformat(timespec="microseconds"),
        (BASE + timedelta(minutes=5)).isoformat(timespec="milliseconds"),
        _iso(10, BASE),
    ]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["rows"] == 5
    assert result["duplicates"] == 2


# ---------------------------------------------------------------------------
# Test G — Empty table: must not crash
# ---------------------------------------------------------------------------

def test_empty_table_does_not_crash(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    # No rows inserted at all.

    result = collector.run_integrity_check()

    assert result["rows"] == 0
    assert result["duplicates"] == 0
    assert result["gap_events"] == 0
    assert result["missing_bars"] == 0
    assert result["issues"] == ["No data found"]


# ---------------------------------------------------------------------------
# Test H — Single candle: must not report a gap
# ---------------------------------------------------------------------------

def test_single_candle_reports_no_gap(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    _insert_rows(collector, [_iso(0, BASE)])

    result = collector.run_integrity_check()

    assert result["rows"] == 1
    assert result["duplicates"] == 0
    assert result["gap_events"] == 0
    assert result["missing_bars"] == 0
    assert result["issues"] == []


# ---------------------------------------------------------------------------
# Interval-awareness: the same absolute gap is measured differently at
# different configured intervals, proving bar_ms (not a hardcoded constant)
# drives the calculation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "interval, minutes, expected_gap_events, expected_missing_bars",
    [
        ("1m", (0, 1, 2, 3), 0, 0),
        ("1m", (0, 1, 3), 1, 1),          # missing minute 2
        ("5m", (0, 5, 10, 15), 0, 0),
        ("15m", (0, 15, 30), 0, 0),
        ("15m", (0, 15, 60), 1, 2),       # missing 00:30, 00:45
        ("1h", (0, 60, 120), 0, 0),
    ],
)
def test_gap_detection_is_interval_aware(
    tmp_path, interval, minutes, expected_gap_events, expected_missing_bars
):
    collector = _make_collector(tmp_path, interval)
    times = [_iso(m, BASE) for m in minutes]
    _insert_rows(collector, times)

    result = collector.run_integrity_check()

    assert result["gap_events"] == expected_gap_events
    assert result["missing_bars"] == expected_missing_bars


# ---------------------------------------------------------------------------
# Only rows for the configured symbol are examined (sanity check that the
# WHERE symbol = ? filter in run_integrity_check still behaves correctly
# alongside the new gap-detection logic).
#
# NOTE: the ohlcv_<interval> table's schema declares
# ``open_time TEXT PRIMARY KEY`` with no symbol column in the key — i.e. the
# primary key is global across the whole table, not scoped per symbol. That
# is a pre-existing schema property (unchanged by this stage; out of scope
# per this task's "no unrelated refactor" instruction — flagged in the
# Stage 2C report) which means two symbols cannot literally share an
# open_time row. This test therefore uses a different, non-colliding time
# range for the second symbol while still proving the WHERE symbol = ?
# filter keeps its rows out of BTCUSDT's integrity result.
# ---------------------------------------------------------------------------

def test_integrity_check_is_scoped_to_configured_symbol(tmp_path):
    collector = _make_collector(tmp_path, "5m")
    # BTCUSDT: clean, no gaps.
    _insert_rows(collector, [_iso(m, BASE) for m in (0, 5, 10)], symbol="BTCUSDT")
    # A different symbol, at a disjoint time range, with a real gap — must
    # not contaminate BTCUSDT's result.
    _insert_rows(collector, [_iso(m, BASE) for m in (1000, 1020)], symbol="ETHUSDT")

    result = collector.run_integrity_check()

    assert result["rows"] == 3
    assert result["gap_events"] == 0
    assert result["missing_bars"] == 0

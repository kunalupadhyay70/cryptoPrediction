"""
DataCollector — upgraded with:
  - Historical pagination (30-90 days of 1-minute OHLCV)
  - Incremental updates (no re-downloading duplicates)
  - Integrity checks: gaps, duplicates, monotonic ordering
  - Clean UTC alignment
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import websockets

from exchange_clients import build_exchange_client
from intervals import interval_to_milliseconds

LOGGER = logging.getLogger(__name__)

# Fixed physical unit conversion (milliseconds in one minute) — used only to
# convert a day-count lookback window into milliseconds. This is NOT the bar
# duration and must stay independent of kline_interval; the actual bar
# duration is computed per-instance in DataCollector.__init__ from the
# configured interval via intervals.interval_to_milliseconds(). Stage 0 found
# the old module-level `BAR_MS = 60_000` being reused for both purposes,
# which silently assumed 1-minute bars everywhere it was used to advance a
# pagination cursor — see DataCollector.bar_ms below for the fix.
_MS_PER_MINUTE = 60_000


@dataclass
class DataCollectorConfig:
    exchange_name: str
    rest_base_url: str
    ws_base_url: str
    symbol: str
    depth_limit: int
    trades_limit: int
    kline_interval: str
    kline_limit: int
    db_path: str
    # Field accepted under both names; target_days takes precedence if supplied
    target_days: int = 30
    days_history: int = 30          # alias kept for backward compat
    pagination_sleep: float = 0.2   # seconds between paginated API calls
    incremental: bool = True
    integrity_check: bool = True

    def __post_init__(self):
        # Resolve alias: if caller passed target_days, mirror it into days_history
        if self.target_days != 30:
            self.days_history = self.target_days
        elif self.days_history != 30:
            self.target_days = self.days_history


class DataCollector:
    def __init__(self, config: DataCollectorConfig, timeout: int = 15) -> None:
        self.config = config
        self.session = requests.Session()
        self.timeout = timeout
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.exchange_client = build_exchange_client(config.exchange_name)
        # Table name follows the interval: ohlcv_1m, ohlcv_5m, etc.
        self.ohlcv_table = f"ohlcv_{config.kline_interval}"
        # Canonical bar duration for THIS collector's configured interval.
        # Computed once here (fails loudly at construction time if
        # kline_interval isn't a supported interval) and reused for every
        # pagination-cursor calculation below instead of the old hardcoded
        # 1-minute assumption.
        self.bar_ms = interval_to_milliseconds(config.kline_interval)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        tbl = self.ohlcv_table
        with self._connect() as conn:
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS data_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT, exchange TEXT, symbol TEXT, note TEXT
                );
                CREATE TABLE IF NOT EXISTS order_book_snapshots (
                    ts TEXT PRIMARY KEY, symbol TEXT,
                    best_bid REAL, best_ask REAL, mid_price REAL, spread REAL,
                    bids_json TEXT, asks_json TEXT
                );
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT,
                    price REAL, qty REAL, is_buyer_maker INTEGER
                );
                CREATE TABLE IF NOT EXISTS {tbl} (
                    open_time TEXT PRIMARY KEY, close_time TEXT,
                    symbol TEXT, interval TEXT,
                    open REAL, high REAL, low REAL, close REAL, volume REAL
                );
                CREATE INDEX IF NOT EXISTS idx_{tbl}_open_time ON {tbl}(open_time);
            """)

    def create_version(self, note: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO data_versions(created_at, exchange, symbol, note) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), self.config.exchange_name, self.config.symbol, note),
            )

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        response = self.session.get(
            f"{self.config.rest_base_url}{path}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _parse_binance_klines(self, klines: List[List[Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "open_time": datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).isoformat(),
                "close_time": datetime.fromtimestamp(row[6] / 1000, tz=timezone.utc).isoformat(),
                "open_time_ms": int(row[0]),
                "open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4]), "volume": float(row[5]),
            }
            for row in klines
        ]

    def _get_newest_open_time_ms(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT MAX(open_time) FROM {self.ohlcv_table} WHERE symbol = ?", (self.config.symbol,)
            ).fetchone()
        if row and row[0]:
            return int(datetime.fromisoformat(row[0]).timestamp() * 1000)
        return None

    def _upsert_klines(self, klines: List[Dict[str, Any]]) -> int:
        inserted = 0
        with self._connect() as conn:
            for k in klines:
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO {self.ohlcv_table} "
                    "(open_time, close_time, symbol, interval, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (k["open_time"], k["close_time"], self.config.symbol, self.config.kline_interval,
                     k["open"], k["high"], k["low"], k["close"], k["volume"]),
                )
                inserted += cur.rowcount
        return inserted

    def collect_historical_paginated(self) -> Dict[str, Any]:
        """
        Fetch target_days of 1m candles via paginated API calls with incremental support.

        Returns a dict with keys:
            total_stored    — total rows now in the DB
            candles_fetched — rows returned by the API this run
            candles_inserted — new rows actually written (duplicates excluded)
            oldest_candle   — ISO timestamp of earliest stored candle
            newest_candle   — ISO timestamp of latest stored candle
            api_calls       — number of HTTP requests made
            integrity       — sub-dict with gap_count, missing_bars,
                              duplicates, span_days, issues list (Stage 2C:
                              gap_count/missing_bars/duplicates are now
                              interval-aware, sourced from
                              run_integrity_check()'s gap_events/
                              missing_bars/duplicates — see that method's
                              docstring for the exact gap-detection formula)
        """
        total_minutes = self.config.target_days * 24 * 60
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        target_start_ms = end_ms - (total_minutes * _MS_PER_MINUTE)

        if self.config.incremental:
            newest_ms = self._get_newest_open_time_ms()
            # Advance past the newest stored candle by exactly one bar of the
            # CONFIGURED interval (was hardcoded to +BAR_MS = +60_000ms,
            # i.e. always "+1 minute" regardless of kline_interval).
            fetch_start_ms = (newest_ms + self.bar_ms) if newest_ms else target_start_ms
            LOGGER.info(
                "Incremental mode: fetching from %s",
                datetime.fromtimestamp(fetch_start_ms / 1000, tz=timezone.utc).isoformat(),
            )
        else:
            fetch_start_ms = target_start_ms

        all_klines: List[Dict] = []
        cursor_ms = fetch_start_ms
        batch_limit = min(self.config.kline_limit, 1500)
        call_count = 0

        while cursor_ms < end_ms:
            try:
                raw = self._get(
                    self.exchange_client.klines_path(),
                    {"symbol": self.config.symbol, "interval": self.config.kline_interval,
                     "startTime": cursor_ms, "limit": batch_limit},
                )
            except requests.RequestException as exc:
                LOGGER.error("Batch fetch failed: %s", exc)
                time.sleep(2)
                continue

            if not raw:
                break

            batch = self._parse_binance_klines(raw)
            all_klines.extend(batch)
            call_count += 1
            # Advance the pagination cursor by exactly one bar of the
            # configured interval (was hardcoded to +BAR_MS = +60_000ms).
            # For non-1m intervals the old code re-requested overlapping
            # windows on every batch, since the next real candle is further
            # away than 60s.
            cursor_ms = raw[-1][0] + self.bar_ms
            LOGGER.info("Batch %d | %d candles | total so far: %d | last: %s",
                        call_count, len(batch), len(all_klines), batch[-1]["open_time"])
            time.sleep(self.config.pagination_sleep)
            if len(raw) < batch_limit:
                break

        inserted = self._upsert_klines(all_klines) if all_klines else 0
        total_stored, oldest, newest = self._db_stats()

        # Integrity check
        #
        # CONTRACT NOTE (Stage 2C): run_integrity_check() now returns a dict
        # (see its docstring) instead of List[str]. "gap_count" here is kept
        # as the existing key name main.py already reads
        # (result["integrity"]["gap_count"]), but its value now comes
        # directly from the interval-aware gap_events count instead of the
        # old approximate "count issue strings containing the word gap"
        # derivation — the old derivation collapsed all gaps in a run into
        # at most one summary string, so gap_count was effectively just 0 or
        # 1 regardless of how many real gaps existed. "missing_bars" and
        # "duplicates" are new keys, added rather than replacing anything.
        integrity_result: Dict[str, Any] = {
            "gap_count": 0, "missing_bars": 0, "duplicates": 0,
            "span_days": 0.0, "issues": [],
        }
        if self.config.integrity_check:
            integrity = self.run_integrity_check()
            issues = integrity["issues"]
            integrity_result["issues"] = issues
            integrity_result["gap_count"] = integrity["gap_events"]
            integrity_result["missing_bars"] = integrity["missing_bars"]
            integrity_result["duplicates"] = integrity["duplicates"]
            if oldest and newest:
                t0 = datetime.fromisoformat(oldest)
                t1 = datetime.fromisoformat(newest)
                integrity_result["span_days"] = (t1 - t0).total_seconds() / 86400
            if not issues:
                LOGGER.info("Integrity check passed")
            else:
                for issue in issues:
                    LOGGER.warning("Integrity: %s", issue)

        LOGGER.info(
            "Collection complete | DB rows: %d | new: %d | %s -> %s",
            total_stored, inserted, oldest, newest,
        )

        return {
            "total_stored": total_stored,
            "candles_fetched": len(all_klines),
            "candles_inserted": inserted,
            "oldest_candle": oldest,
            "newest_candle": newest,
            "api_calls": call_count,
            "integrity": integrity_result,
        }

    def collect_orderbook_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Fetch a single order-book + trades snapshot and persist to DB.
        Called separately from kline collection so live mode can refresh
        microstructure without re-fetching all OHLCV data.
        """
        try:
            depth = self._get(
                self.exchange_client.depth_path(),
                {"symbol": self.config.symbol, "limit": self.config.depth_limit},
            )
            trades_raw = self._get(
                self.exchange_client.trades_path(),
                {"symbol": self.config.symbol, "limit": self.config.trades_limit},
            )
        except requests.RequestException as exc:
            LOGGER.error("Orderbook snapshot failed: %s", exc)
            return None

        bids = depth["bids"]
        asks = depth["asks"]
        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        ts = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO order_book_snapshots "
                "(ts, symbol, best_bid, best_ask, mid_price, spread, bids_json, asks_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, self.config.symbol, best_bid, best_ask, mid_price, spread,
                 json.dumps(bids), json.dumps(asks)),
            )
            for trade in trades_raw:
                conn.execute(
                    "INSERT OR IGNORE INTO trades "
                    "(trade_id, ts, symbol, price, qty, is_buyer_maker) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(trade.get("id", trade.get("t", "unknown"))),
                        datetime.fromtimestamp(
                            trade.get("time", trade.get("T", 0)) / 1000, tz=timezone.utc
                        ).isoformat(),
                        self.config.symbol,
                        float(trade.get("price", trade.get("p", 0.0))),
                        float(trade.get("qty", trade.get("q", 0.0))),
                        int(bool(trade.get("isBuyerMaker", trade.get("m", False)))),
                    ),
                )

        LOGGER.info("Orderbook snapshot stored | mid=%.2f spread=%.6f", mid_price, spread)
        return {"ts": ts, "mid_price": mid_price, "spread": spread}

    def collect_rest_once(self) -> Optional[Dict[str, Any]]:
        """Fetch latest candles + orderbook in one shot (used by live loop)."""
        try:
            klines_raw = self._get(
                self.exchange_client.klines_path(),
                {"symbol": self.config.symbol, "interval": self.config.kline_interval, "limit": 10},
            )
        except requests.RequestException as exc:
            LOGGER.error("REST collection failed: %s", exc)
            return None
        klines = self._parse_binance_klines(klines_raw)
        self._upsert_klines(klines)
        snap = self.collect_orderbook_snapshot()
        return snap

    def _db_stats(self) -> Tuple[int, Optional[str], Optional[str]]:
        with self._connect() as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {self.ohlcv_table} WHERE symbol = ?", (self.config.symbol,)
            ).fetchone()[0]
            row = conn.execute(
                f"SELECT MIN(open_time), MAX(open_time) FROM {self.ohlcv_table} WHERE symbol = ?",
                (self.config.symbol,),
            ).fetchone()
        return count, (row[0] if row else None), (row[1] if row else None)

    def run_integrity_check(self) -> Dict[str, Any]:
        """Validate OHLCV integrity for this collector's table.

        Interval-aware gap detection (Stage 2C fix): expected candle spacing
        is derived from ``self.bar_ms`` (the Stage 2A canonical interval
        utility, computed once in ``__init__`` from the configured
        ``kline_interval``), NOT a hardcoded "> 2 minutes" threshold. The
        previous hardcoded threshold implicitly assumed ~1-minute bars —
        for a configured 5m interval, consecutive valid candles are 5
        minutes apart, so the old ">2 minute" rule flagged essentially every
        adjacent pair as a gap (Stage 0 reproduced this: 19,999/20,000 valid
        synthetic 5-minute bars were incorrectly flagged as gaps).

        Ordering: rows are read via ``ORDER BY open_time ASC``, so the gap
        scan below always walks candles in chronological order as stored.
        This does not detect whether rows were *inserted* out of order
        (SQLite has no notion of insertion order independent of the primary
        key here) — only whether, once sorted, the sequence of timestamps is
        internally consistent (see the "Non-monotonic" check below, which is
        a defensive no-op today precisely because the query itself already
        sorts, but is kept in case a future caller feeds in an unsorted
        list).

        Returns a dict (CONTRACT CHANGE — see note at the end):
            rows          — total row count examined for this symbol
            duplicates    — count of duplicate ``open_time`` values (extra
                             occurrences beyond the first of each timestamp)
            gap_events    — count of individual gaps, i.e. consecutive-row
                             deltas strictly greater than one configured bar
                             duration (``self.bar_ms``)
            missing_bars  — total number of missing candles summed across
                             all gap events; for one gap with
                             ``delta = current_open_time - previous_open_time``,
                             the missing-bar count is
                             ``(delta // self.bar_ms) - 1`` (e.g. for a 5m
                             interval, 00:00 -> 00:15 has delta=15m,
                             missing = (15m // 5m) - 1 = 2, namely 00:05 and
                             00:10)
            issues        — human-readable issue strings, kept for logging

        CONTRACT CHANGE (Stage 2C): this method previously returned
        ``List[str]`` (just the ``issues`` messages). It now returns the
        dict above. A repo-wide search found exactly one caller —
        ``collect_historical_paginated`` in this same module — which has
        been updated accordingly in this change. No other module currently
        calls ``run_integrity_check`` directly.
        """
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT open_time FROM {self.ohlcv_table} WHERE symbol = ? ORDER BY open_time ASC",
                (self.config.symbol,),
            ).fetchall()

        if not rows:
            return {
                "rows": 0,
                "duplicates": 0,
                "gap_events": 0,
                "missing_bars": 0,
                "issues": ["No data found"],
            }

        times = [r[0] for r in rows]
        times_ms = [int(datetime.fromisoformat(t).timestamp() * 1000) for t in times]

        issues: List[str] = []

        # Duplicate detection: count occurrences beyond the first of each
        # timestamp. Duplicates are reported, never silently dropped before
        # the gap scan below (a duplicate produces delta == 0 between the
        # repeated rows, which the gap scan below correctly treats as
        # neither a gap nor a non-monotonic error).
        seen = set()
        duplicates = 0
        for t in times_ms:
            if t in seen:
                duplicates += 1
            else:
                seen.add(t)
        if duplicates:
            issues.append(f"Found {duplicates} duplicate timestamp(s)")

        gap_events = 0
        missing_bars = 0
        for i in range(1, len(times_ms)):
            delta = times_ms[i] - times_ms[i - 1]
            if delta < 0:
                issues.append(f"Non-monotonic timestamp near {times[i]}")
                continue
            if delta == 0:
                # Duplicate row — already counted above, not a gap.
                continue
            if delta > self.bar_ms:
                missing = (delta // self.bar_ms) - 1
                gap_events += 1
                missing_bars += missing

        if gap_events:
            issues.append(
                f"Found {gap_events} gap event(s) totaling {missing_bars} missing "
                f"{self.config.kline_interval} bar(s)"
            )

        return {
            "rows": len(times_ms),
            "duplicates": duplicates,
            "gap_events": gap_events,
            "missing_bars": missing_bars,
            "issues": issues,
        }

    def load_ohlcv_dataframe(self):
        """Load this collector's stored OHLCV rows for its configured
        symbol as a pandas DataFrame, sorted ascending by open_time, with
        columns [open_time, open, high, low, close, volume] -- open_time
        parsed to a tz-aware (UTC) pandas Timestamp column, matching the
        shape target_engineering/dataset_builder expect (Stage 2P: the
        single loader used to hand off from storage to the new causal
        feature/target pipeline; no other module re-implements this
        query).
        """
        import pandas as pd

        with self._connect() as conn:
            df = pd.read_sql_query(
                f"SELECT open_time, open, high, low, close, volume "
                f"FROM {self.ohlcv_table} WHERE symbol = ? ORDER BY open_time ASC",
                conn, params=(self.config.symbol,),
            )
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df

    async def collect_orderbook_ws(self, seconds: int) -> None:
        end = time.time() + seconds
        async with websockets.connect(self.config.ws_base_url, ping_interval=20) as ws:
            await ws.send(json.dumps(
                self.exchange_client.depth_ws_payload(self.config.symbol, self.config.depth_limit)
            ))
            await ws.send(json.dumps(self.exchange_client.trades_ws_payload(self.config.symbol)))
            while time.time() < end:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(message)
                LOGGER.debug("WS tick: %s", list(data.keys()) if isinstance(data, dict) else type(data))
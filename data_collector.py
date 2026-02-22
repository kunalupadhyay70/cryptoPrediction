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

LOGGER = logging.getLogger(__name__)
BAR_MS = 60_000  # 1-minute bar in ms


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
            integrity       — sub-dict with gap_count, span_days, issues list
        """
        total_minutes = self.config.target_days * 24 * 60
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        target_start_ms = end_ms - (total_minutes * BAR_MS)

        if self.config.incremental:
            newest_ms = self._get_newest_open_time_ms()
            fetch_start_ms = (newest_ms + BAR_MS) if newest_ms else target_start_ms
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
            cursor_ms = raw[-1][0] + BAR_MS
            LOGGER.info("Batch %d | %d candles | total so far: %d | last: %s",
                        call_count, len(batch), len(all_klines), batch[-1]["open_time"])
            time.sleep(self.config.pagination_sleep)
            if len(raw) < batch_limit:
                break

        inserted = self._upsert_klines(all_klines) if all_klines else 0
        total_stored, oldest, newest = self._db_stats()

        # Integrity check
        integrity_result: Dict[str, Any] = {"gap_count": 0, "span_days": 0.0, "issues": []}
        if self.config.integrity_check:
            issues = self.run_integrity_check()
            integrity_result["issues"] = issues
            integrity_result["gap_count"] = sum(1 for i in issues if "gap" in i.lower())
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

    def run_integrity_check(self) -> List[str]:
        issues: List[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT open_time FROM {self.ohlcv_table} WHERE symbol = ? ORDER BY open_time ASC",
                (self.config.symbol,),
            ).fetchall()
        if not rows:
            return ["No data found"]
        times = [r[0] for r in rows]
        if len(times) != len(set(times)):
            issues.append(f"Found {len(times) - len(set(times))} duplicate timestamps")
        gap_count = 0
        prev_dt = datetime.fromisoformat(times[0])
        for ts in times[1:]:
            curr_dt = datetime.fromisoformat(ts)
            diff = (curr_dt - prev_dt).total_seconds() / 60
            if diff < 0:
                issues.append(f"Non-monotonic timestamp near {ts}")
            elif diff > 2:
                gap_count += 1
            prev_dt = curr_dt
        if gap_count > 0:
            issues.append(f"Found {gap_count} gaps > 2 minutes (likely exchange downtime)")
        return issues

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
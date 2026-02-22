"""Exchange client adapters for Binance/Delta (REST + websocket endpoint metadata)."""

from typing import Dict, Protocol


class ExchangeClient(Protocol):
    def depth_path(self) -> str: ...
    def trades_path(self) -> str: ...
    def klines_path(self) -> str: ...
    def depth_ws_payload(self, symbol: str, levels: int) -> Dict: ...
    def trades_ws_payload(self, symbol: str) -> Dict: ...


class BinanceFuturesClient:
    def depth_path(self) -> str:
        return "/fapi/v1/depth"

    def trades_path(self) -> str:
        return "/fapi/v1/trades"

    def klines_path(self) -> str:
        return "/fapi/v1/klines"

    def depth_ws_payload(self, symbol: str, levels: int) -> Dict:
        return {"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@depth{levels}@100ms"], "id": 1}

    def trades_ws_payload(self, symbol: str) -> Dict:
        return {"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@aggTrade"], "id": 2}


class DeltaFuturesClient:
    def depth_path(self) -> str:
        return "/v2/l2orderbook"

    def trades_path(self) -> str:
        return "/v2/history/trades"

    def klines_path(self) -> str:
        return "/v2/history/candles"

    def depth_ws_payload(self, symbol: str, levels: int) -> Dict:
        return {"type": "subscribe", "payload": {"channels": [{"name": "l2_orderbook", "symbols": [symbol]}]}}

    def trades_ws_payload(self, symbol: str) -> Dict:
        return {"type": "subscribe", "payload": {"channels": [{"name": "all_trades", "symbols": [symbol]}]}}


def build_exchange_client(name: str):
    if name == "delta_futures":
        return DeltaFuturesClient()
    return BinanceFuturesClient()

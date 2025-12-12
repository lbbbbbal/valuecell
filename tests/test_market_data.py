import httpx
import pytest
from typing import Any, Dict, List

from valuecell_ext.binance_market_data import (
    BinanceMarketData,
    Candle,
    IntervalBlock,
)


class DummyResponse:
    def __init__(self, status_code: int, json_data: Any, headers: Dict[str, str] | None = None):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=httpx.Response(self.status_code))


class DummyClient:
    def __init__(self, responses: Dict[str, DummyResponse]):
        self.responses = responses
        self.calls: List[str] = []

    async def get(self, endpoint: str, params: Dict[str, Any], timeout: float) -> DummyResponse:  # type: ignore
        self.calls.append(endpoint)
        key = (endpoint, tuple(sorted(params.items())))
        return self.responses.get(endpoint) or self.responses.get(str(key)) or DummyResponse(404, {})

    async def aclose(self) -> None:  # pragma: no cover - not used
        return None


@pytest.mark.asyncio
async def test_normalize_symbol():
    md = BinanceMarketData(client=DummyClient({}))
    assert md.normalize_symbol("xrp/usdt") == "XRPUSDT"
    assert md.normalize_symbol("XRP/USDT:USDT") == "XRPUSDT"


@pytest.mark.asyncio
async def test_fapi_kline_parsing():
    klines = [
        [1690000000000, "1", "2", "0.5", "1.5", "10", 0, 0, 5, 2, 3],
    ]
    responses = {"/fapi/v1/klines": DummyResponse(200, klines)}
    md = BinanceMarketData(client=DummyClient(responses))
    block = await md.get_candles("BTC/USDT", "1m", limit=1)
    assert isinstance(block, IntervalBlock)
    assert len(block.candles) == 1
    candle = block.candles[0]
    assert candle.ts_ms == 1690000000000
    assert candle.taker_buy_quote == 3.0


class CCXTDummy:
    def __init__(self, market_type: str = "future", markets: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.market_type = market_type
        self.closed = False
        self._markets = markets
        self.last_symbol: str | None = None

    async def load_markets(self) -> Dict[str, Dict[str, Any]]:
        return self._markets or {"BTC/USDT:USDT": {"type": self.market_type, "linear": True, "symbol": "BTC/USDT:USDT"}}

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: Any, limit: int, params: Dict[str, Any]) -> List[List[Any]]:
        self.last_symbol = symbol
        return [[1690000000000, 1, 2, 0.5, 1.5, 10]]

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ccxt_fallback_triggered():
    async def failing_request(self, endpoint: str, params: Dict[str, Any] | None = None):  # type: ignore
        raise httpx.RequestError("boom")

    responses: Dict[str, DummyResponse] = {}
    md = BinanceMarketData(client=DummyClient(responses), ccxt_client=CCXTDummy())
    md._request = failing_request.__get__(md)  # type: ignore
    block = await md.get_candles("BTC/USDT", "1m", limit=1)
    assert block.source == "ccxt"


@pytest.mark.asyncio
async def test_ccxt_rejects_spot():
    md = BinanceMarketData(client=DummyClient({}), ccxt_client=CCXTDummy(market_type="spot"))
    with pytest.raises(RuntimeError):
        await md._fetch_ccxt_klines("BTCUSDT", "1m", 1, None, None)


@pytest.mark.asyncio
async def test_ccxt_resolves_busd_symbol():
    markets = {
        "BTC/BUSD:BUSD": {"type": "future", "linear": True, "symbol": "BTC/BUSD:BUSD"},
        "BTC/USDT:USDT": {"type": "future", "linear": True, "symbol": "BTC/USDT:USDT"},
    }
    md = BinanceMarketData(client=DummyClient({}), ccxt_client=CCXTDummy(markets=markets))
    block = await md._fetch_ccxt_klines("BTCBUSD", "1m", 1, None, None)
    assert len(block) == 1
    assert md.ccxt_client.last_symbol == "BTC/BUSD:BUSD"


@pytest.mark.asyncio
async def test_resample_and_coverage():
    candles = []
    start = 1690000000000
    for i in range(15):
        candles.append(Candle(ts_ms=start + i * 60000, open=1 + i, high=2 + i, low=0.5 + i, close=1.5 + i, volume=1))
    md = BinanceMarketData(client=DummyClient({}))
    resampled, coverage = md._resample_from_1m(candles, "15m")
    assert len(resampled) == 1
    res = resampled[0]
    assert res.open == 1
    assert res.close == candles[-1].close
    assert res.high == max(c.high for c in candles)
    assert coverage == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_insufficient_coverage_marks_missing():
    candles = []
    start = 1690000000000
    for i in range(5):
        candles.append(Candle(ts_ms=start + i * 60000, open=1, high=1, low=1, close=1, volume=1))
    md = BinanceMarketData(client=DummyClient({}))
    block = md._build_block("15m", candles, "resampled", coverage=0.2)
    assert block.missing is True


@pytest.mark.asyncio
async def test_exchangeinfo_cache_ttl():
    responses = {"/fapi/v1/exchangeInfo": DummyResponse(200, {"symbols": []})}
    md = BinanceMarketData(client=DummyClient(responses))
    first = await md.get_exchange_info()
    second = await md.get_exchange_info()
    assert first is second


@pytest.mark.asyncio
async def test_edge_floor_bps_math():
    micro_resp = DummyResponse(200, {"bidPrice": "100", "askPrice": "100.1"})
    md = BinanceMarketData(client=DummyClient({"/fapi/v1/ticker/bookTicker": micro_resp}))
    micro = await md.get_micro("BTC/USDT")
    spread_bps = (100.1 - 100) / 100.05 * 1e4
    expected_edge = (2 * md.config.taker_fee_bps + spread_bps + max(spread_bps, md.config.slippage_floor_bps))
    assert micro.edge_floor_bps == pytest.approx(expected_edge)

import asyncio
from typing import Any, List

import pytest

from valuecell.agents.common.trading.data.fallback_candles import FuturesCandleFetcher
from valuecell.agents.common.trading.models import Candle, InstrumentRef


class _MockExchange:
    def __init__(self, payload: List[List[Any]] | Exception):
        self._payload = payload

    async def fetch_ohlcv(self, symbol: str, interval: str, since: Any, limit: int):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.mark.asyncio
async def test_rest_failure_falls_back_to_ccxt():
    exchange = _MockExchange(payload=[[0, 1, 2, 3, 4, 5]])
    fetcher = FuturesCandleFetcher(ccxt_client=exchange, timeout_s=0.1)

    async def _fail_rest(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")

    fetcher._fetch_via_rest = _fail_rest  # type: ignore

    result = await fetcher.fetch(symbols=["BTCUSDT"], interval="1m", lookback=1)
    assert result.meta.interval_source == "ccxt"
    assert result.candles[0].close == 4


@pytest.mark.asyncio
async def test_resample_used_when_rest_and_ccxt_fail():
    fetcher = FuturesCandleFetcher(ccxt_client=None, timeout_s=0.1)
    one_minute_rows = [
        [0, 1, 2, 0.5, 1.5, 10],
        [60_000, 1.5, 2.5, 1.0, 2.0, 12],
        [120_000, 2.0, 3.0, 1.5, 2.5, 8],
        [180_000, 2.5, 4.0, 2.0, 3.0, 7],
    ]

    async def _rest_stub(symbols, interval, lookback):  # noqa: ANN001
        if interval == "1m":
            candles = []
            for row in one_minute_rows:
                ts, o, h, l, c, v = row
                candles.append(
                    Candle(
                        ts=ts,
                        instrument=InstrumentRef(symbol="BTCUSDT", exchange_id="binance"),
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=v,
                        interval=interval,
                    )
                )
            return candles
        raise RuntimeError("boom")

    fetcher._fetch_via_rest = _rest_stub  # type: ignore

    async def _fail_ccxt(*_a, **_k):  # noqa: ANN001
        raise RuntimeError("fail")

    fetcher._fetch_via_ccxt = _fail_ccxt  # type: ignore

    result = await fetcher.fetch(symbols=["BTCUSDT"], interval="2m", lookback=2)
    assert result.meta.interval_source == "resample"
    assert result.meta.interval_confidence == "low"
    assert len(result.candles) == 2
    assert result.candles[0].open == 1
    assert result.candles[0].high == 2.5
    assert result.candles[0].low == 0.5
    assert result.candles[0].close == 2.0

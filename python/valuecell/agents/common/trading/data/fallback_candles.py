from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List

import httpx
from loguru import logger

from valuecell.agents.common.trading.models import Candle, InstrumentRef

BINANCE_FAPI_URL = "https://fapi.binance.com"


@dataclass
class CandleFetchMeta:
    interval_source: str
    interval_confidence: str


@dataclass
class CandleFetchResult:
    candles: List[Candle]
    meta: CandleFetchMeta


class FuturesCandleFetcher:
    """Fetch candles with REST->ccxt->resample fallbacks for USD-M futures."""

    def __init__(self, *, ccxt_client: Optional[object] = None, timeout_s: float = 10.0):
        self._ccxt_client = ccxt_client
        self._timeout_s = timeout_s

    async def fetch(
        self, *, symbols: Iterable[str], interval: str, lookback: int
    ) -> CandleFetchResult:
        candles: List[Candle] = []
        try:
            candles = await self._fetch_via_rest(symbols, interval, lookback)
            if candles:
                return CandleFetchResult(
                    candles=candles,
                    meta=CandleFetchMeta(
                        interval_source="fapi_rest", interval_confidence="high"
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Binance REST klines failed: {}", exc)

        try:
            candles = await self._fetch_via_ccxt(symbols, interval, lookback)
            if candles:
                return CandleFetchResult(
                    candles=candles,
                    meta=CandleFetchMeta(
                        interval_source="ccxt", interval_confidence="medium"
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CCXT klines fallback failed: {}", exc)

        resampled = await self._resample_from_one_minute(symbols, interval, lookback)
        return CandleFetchResult(
            candles=resampled,
            meta=CandleFetchMeta(
                interval_source="resample", interval_confidence="low"
            ),
        )

    async def _fetch_via_rest(
        self, symbols: Iterable[str], interval: str, lookback: int
    ) -> List[Candle]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            tasks = [
                self._fetch_symbol_rest(client, symbol, interval, lookback)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks)
            return [c for sub in results for c in sub]

    async def _fetch_symbol_rest(
        self, client: httpx.AsyncClient, symbol: str, interval: str, lookback: int
    ) -> List[Candle]:
        params = {"symbol": symbol.replace("-", ""), "interval": interval, "limit": lookback}
        resp = await client.get(f"{BINANCE_FAPI_URL}/fapi/v1/klines", params=params)
        resp.raise_for_status()
        raw = resp.json()
        candles: List[Candle] = []
        for row in raw:
            ts, open_px, high, low, close, volume = row[:6]
            candles.append(
                Candle(
                    ts=int(ts),
                    instrument=InstrumentRef(symbol=symbol, exchange_id="binance"),
                    open=float(open_px),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                    interval=interval,
                )
            )
        return candles

    async def _fetch_via_ccxt(
        self, symbols: Iterable[str], interval: str, lookback: int
    ) -> List[Candle]:
        if self._ccxt_client is None:
            return []
        tasks = [
            self._ccxt_client.fetch_ohlcv(symbol.replace("-", "/"), interval, None, lookback)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)
        candles: List[Candle] = []
        for symbol, rows in zip(symbols, results):
            for row in rows:
                ts, open_px, high, low, close, volume = row[:6]
                candles.append(
                    Candle(
                        ts=int(ts),
                        instrument=InstrumentRef(symbol=symbol, exchange_id="binance"),
                        open=float(open_px),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume),
                        interval=interval,
                    )
                )
        return candles

    async def _resample_from_one_minute(
        self, symbols: Iterable[str], interval: str, lookback: int
    ) -> List[Candle]:
        base_interval = "1m"
        one_minute = await self._fetch_via_rest(symbols, base_interval, max(lookback, 10))
        grouped: dict[str, List[Candle]] = {}
        for candle in one_minute:
            grouped.setdefault(candle.instrument.symbol, []).append(candle)
        resampled: List[Candle] = []
        bucket_minutes = self._interval_to_minutes(interval)
        for symbol, rows in grouped.items():
            rows_sorted = sorted(rows, key=lambda c: c.ts)
            for i in range(0, len(rows_sorted), bucket_minutes):
                bucket = rows_sorted[i : i + bucket_minutes]
                if len(bucket) < bucket_minutes:
                    continue
                resampled.append(
                    Candle(
                        ts=bucket[-1].ts,
                        instrument=bucket[-1].instrument,
                        open=bucket[0].open,
                        high=max(c.high for c in bucket),
                        low=min(c.low for c in bucket),
                        close=bucket[-1].close,
                        volume=sum(c.volume for c in bucket),
                        interval=interval,
                    )
                )
        return resampled

    def _interval_to_minutes(self, interval: str) -> int:
        suffix = interval[-1]
        value = int(interval[:-1])
        if suffix == "m":
            return value
        if suffix == "h":
            return value * 60
        raise ValueError(f"Unsupported interval for resampling: {interval}")

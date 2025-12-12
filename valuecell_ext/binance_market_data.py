from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
from loguru import logger

try:
    import ccxt.async_support as ccxt_async
except Exception:  # pragma: no cover - optional dependency
    ccxt_async = None

from valuecell_ext.rate_limiter import EndpointRateLimiter

FAPI_BASE_URL = "https://fapi.binance.com"


@dataclass
class Candle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[float] = None
    taker_buy_base: Optional[float] = None
    taker_buy_quote: Optional[float] = None


@dataclass
class MarketMicro:
    bid: float
    ask: float
    mid: float
    spread_bps: float
    estimated_fee_bps: float
    estimated_slippage_bps: float
    edge_floor_bps: float


@dataclass
class Funding:
    mark_price: float
    funding_rate: float
    next_funding_time_ms: int


@dataclass
class IntervalBlock:
    interval: str
    candles: List[Candle]
    source: str
    missing: bool
    coverage: float


@dataclass
class MarketDataConfig:
    request_timeout_s: float = 8.0
    retries: int = 2
    retry_backoff_s: float = 0.75
    ccxt_enabled: bool = True
    taker_fee_bps: float = 7.0  # 0.07%
    maker_fee_bps: float = 2.0
    slippage_floor_bps: float = 1.0
    edge_mult: float = 1.0
    exchangeinfo_ttl_s: int = 3600
    cooldown_failures: int = 3
    cooldown_window_s: int = 60
    resample_coverage_threshold: float = 0.85
    expected_windows: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.expected_windows is None:
            self.expected_windows = {"1m": 120, "15m": 120, "1h": 120}


class FailureTracker:
    def __init__(self) -> None:
        self.failures: Dict[Tuple[str, str, str], List[float]] = {}

    def record(self, symbol: str, interval: str, layer: str) -> None:
        key = (symbol, interval, layer)
        now = time.monotonic()
        history = self.failures.setdefault(key, [])
        history.append(now)
        self.failures[key] = [t for t in history if now - t <= 60]

    def should_skip(self, symbol: str, interval: str, layer: str, threshold: int) -> bool:
        key = (symbol, interval, layer)
        now = time.monotonic()
        history = [t for t in self.failures.get(key, []) if now - t <= 60]
        self.failures[key] = history
        return len(history) >= threshold


class BinanceMarketData:
    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[MarketDataConfig] = None,
        limiter: Optional[EndpointRateLimiter] = None,
        ccxt_client: Optional[object] = None,
    ) -> None:
        self.config = config or MarketDataConfig()
        self.client = client or httpx.AsyncClient(base_url=FAPI_BASE_URL)
        self.limiter = limiter or EndpointRateLimiter(default_rate=1200, capacities={"klines": 1500})
        self.ccxt_client = ccxt_client
        self._exchangeinfo_cache: Optional[Tuple[float, dict]] = None
        self.failures = FailureTracker()
        self._stats: Dict[str, int] = {"fapi": 0, "ccxt": 0, "resampled": 0, "missing": 0}

    async def close(self) -> None:
        await self.client.aclose()
        if self.ccxt_client is not None and hasattr(self.ccxt_client, "close"):
            await self.ccxt_client.close()

    async def _request(self, endpoint: str, params: Optional[dict] = None) -> httpx.Response:
        params = params or {}
        backoff = self.config.retry_backoff_s
        for attempt in range(self.config.retries + 1):
            success = await self.limiter.acquire(endpoint)
            if not success:
                raise httpx.HTTPError("Rate limit exceeded locally")
            try:
                resp = await self.client.get(
                    endpoint,
                    params=params,
                    timeout=self.config.request_timeout_s,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    logger.warning("Hit Binance 429, cooling down for {s}s", s=retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if 500 <= exc.response.status_code < 600 and attempt < self.config.retries:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                raise
            except (httpx.RequestError, httpx.ReadTimeout):
                if attempt < self.config.retries:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                raise
        raise httpx.HTTPError("Unreachable")

    async def get_server_time(self) -> int:
        resp = await self._request("/fapi/v1/time")
        data = resp.json()
        return int(data["serverTime"])

    async def get_exchange_info(self, force_refresh: bool = False) -> dict:
        now = time.monotonic()
        if not force_refresh and self._exchangeinfo_cache:
            ts, cached = self._exchangeinfo_cache
            if now - ts < self.config.exchangeinfo_ttl_s:
                return cached
        try:
            resp = await self._request("/fapi/v1/exchangeInfo")
            data = resp.json()
            self._exchangeinfo_cache = (now, data)
            return data
        except Exception:
            logger.warning("Failed to refresh exchangeInfo, using stale cache if available")
            if self._exchangeinfo_cache:
                return self._exchangeinfo_cache[1]
            raise

    def clear_exchangeinfo_cache(self) -> None:
        self._exchangeinfo_cache = None

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        upper = symbol.upper()
        if ":" in upper:
            upper = upper.split(":", 1)[0]
        normalized = upper.replace("/", "")
        if normalized.endswith("USDTUSDT"):
            normalized = normalized[:-4]
        return normalized

    @staticmethod
    def _parse_kline_row(row: Iterable) -> Candle:
        ts_ms = int(row[0])
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])
        trades = float(row[8]) if len(row) > 8 else None
        taker_buy_base = float(row[9]) if len(row) > 9 else None
        taker_buy_quote = float(row[10]) if len(row) > 10 else None
        return Candle(
            ts_ms=ts_ms,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            trades=trades,
            taker_buy_base=taker_buy_base,
            taker_buy_quote=taker_buy_quote,
        )

    async def _fetch_fapi_klines(
        self, symbol: str, interval: str, limit: int, start: Optional[int], end: Optional[int]
    ) -> List[Candle]:
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end
        resp = await self._request("/fapi/v1/klines", params=params)
        data = resp.json()
        candles = [self._parse_kline_row(row) for row in data if None not in row[:6]]
        return candles

    async def _fetch_ccxt_klines(
        self, symbol: str, interval: str, limit: int, start: Optional[int], end: Optional[int]
    ) -> List[Candle]:
        if not self.config.ccxt_enabled or ccxt_async is None:
            raise RuntimeError("ccxt not available")
        if self.ccxt_client is None:
            self.ccxt_client = ccxt_async.binance({
                "options": {"defaultType": "future"},
                "enableRateLimit": True,
            })
        market_symbol = symbol.replace("USDT", "/USDT")
        resolved_symbol = f"{market_symbol}:USDT"
        markets = await self.ccxt_client.load_markets()
        market = markets.get(resolved_symbol) or markets.get(market_symbol)
        logger.debug("CCXT market resolved {sym} -> {market}", sym=symbol, market=market)
        if not market or market.get("type") != "future" or market.get("linear") is not True:
            raise RuntimeError("CCXT market type mismatch")
        ohlcv = await self.ccxt_client.fetch_ohlcv(resolved_symbol, timeframe=interval, since=start, limit=limit, params={})
        candles = []
        for row in ohlcv:
            if None in row[:6]:
                continue
            candles.append(
                Candle(
                    ts_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return candles

    def _record_failure(self, symbol: str, interval: str, layer: str) -> None:
        self.failures.record(symbol, interval, layer)

    def _should_skip_layer(self, symbol: str, interval: str, layer: str) -> bool:
        return self.failures.should_skip(symbol, interval, layer, self.config.cooldown_failures)

    def _log_fetch(self, symbol: str, interval: str, source: str, count: int, start_ts: float, reason: str) -> None:
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        logger.info(
            "market_data fetch symbol={symbol} interval={interval} source={source} candles={count} duration_ms={duration} reason={reason}",
            symbol=symbol,
            interval=interval,
            source=source,
            count=count,
            duration=duration_ms,
            reason=reason,
        )

    def _update_stats(self, source: str) -> None:
        if source in self._stats:
            self._stats[source] += 1
        total = sum(self._stats.values())
        if total >= 10:
            degraded = (self._stats.get("resampled", 0) + self._stats.get("missing", 0)) / total
            if degraded > 0.4:
                logger.warning("High degraded data ratio: {ratio:.2%}", ratio=degraded)
            self._stats = {k: 0 for k in self._stats}

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        allow_resample: bool = True,
    ) -> IntervalBlock:
        norm_symbol = self.normalize_symbol(symbol)
        reason = ""
        start_ts = time.monotonic()

        if not self._should_skip_layer(norm_symbol, interval, "fapi"):
            try:
                candles = await self._fetch_fapi_klines(norm_symbol, interval, limit, start, end)
                block = self._build_block(interval, candles, "fapi")
                self._log_fetch(norm_symbol, interval, "fapi", len(block.candles), start_ts, reason)
                self._update_stats("fapi")
                return block
            except Exception as exc:
                reason = f"fapi_failed:{exc}"
                self._record_failure(norm_symbol, interval, "fapi")

        if not self._should_skip_layer(norm_symbol, interval, "ccxt"):
            try:
                candles = await self._fetch_ccxt_klines(norm_symbol, interval, limit, start, end)
                block = self._build_block(interval, candles, "ccxt")
                self._log_fetch(norm_symbol, interval, "ccxt", len(block.candles), start_ts, reason)
                self._update_stats("ccxt")
                return block
            except Exception as exc:
                reason = f"ccxt_failed:{exc}"
                self._record_failure(norm_symbol, interval, "ccxt")

        if interval in ("15m", "1h") and allow_resample:
            base_block = await self.get_candles(
                norm_symbol,
                "1m",
                limit=limit * (60 if interval == "1h" else 15),
                start=start,
                end=end,
                allow_resample=False,
            )
            resampled, coverage = self._resample_from_1m(base_block.candles, interval)
            block = self._build_block(interval, resampled, "resampled", coverage=coverage)
            self._log_fetch(norm_symbol, interval, "resampled", len(block.candles), start_ts, reason)
            self._update_stats("resampled")
            return block

        block = IntervalBlock(interval=interval, candles=[], source="missing", missing=True, coverage=0.0)
        self._log_fetch(norm_symbol, interval, "missing", 0, start_ts, reason)
        self._update_stats("missing")
        return block

    def _build_block(self, interval: str, candles: List[Candle], source: str, coverage: Optional[float] = None) -> IntervalBlock:
        missing = False
        required = self.config.expected_windows.get(interval, 0)
        if len(candles) < required:
            missing = True
        if coverage is None:
            coverage = 1.0 if candles else 0.0
        if coverage < self.config.resample_coverage_threshold:
            missing = True
        return IntervalBlock(interval=interval, candles=candles, source=source, missing=missing, coverage=coverage)

    @staticmethod
    def _interval_minutes(interval: str) -> int:
        if interval.endswith("m"):
            return int(interval[:-1])
        if interval.endswith("h"):
            return int(interval[:-1]) * 60
        raise ValueError(f"Unsupported interval {interval}")

    def _resample_from_1m(self, candles: List[Candle], target_interval: str) -> Tuple[List[Candle], float]:
        if not candles:
            return [], 0.0
        minutes = self._interval_minutes(target_interval)
        base_ts = candles[0].ts_ms
        grouped: Dict[int, List[Candle]] = {}
        for candle in candles:
            bucket = (candle.ts_ms - base_ts) // (minutes * 60000)
            grouped.setdefault(bucket, []).append(candle)
        resampled: List[Candle] = []
        kept = 0
        for _, bucket_candles in sorted(grouped.items()):
            if not bucket_candles:
                continue
            expected = minutes
            if len(bucket_candles) < expected * self.config.resample_coverage_threshold:
                continue
            bucket_candles = sorted(bucket_candles, key=lambda c: c.ts_ms)
            open_price = bucket_candles[0].open
            close = bucket_candles[-1].close
            high = max(c.high for c in bucket_candles)
            low = min(c.low for c in bucket_candles)
            volume = sum(c.volume for c in bucket_candles)
            resampled.append(
                Candle(
                    ts_ms=bucket_candles[0].ts_ms,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
            kept += 1
        total = len(grouped)
        coverage = kept / total if total else 0.0
        return resampled, coverage

    def _edge_floor(self, micro: MarketMicro) -> float:
        return micro.edge_floor_bps

    async def get_micro(self, symbol: str) -> MarketMicro:
        norm_symbol = self.normalize_symbol(symbol)
        resp = await self._request("/fapi/v1/ticker/bookTicker", params={"symbol": norm_symbol})
        data = resp.json()
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 1e4 if mid else 0.0
        fee_bps = self.config.taker_fee_bps
        slippage_bps = max(spread_bps, self.config.slippage_floor_bps)
        edge_floor_bps = (2 * fee_bps + spread_bps + slippage_bps) * self.config.edge_mult
        return MarketMicro(
            bid=bid,
            ask=ask,
            mid=mid,
            spread_bps=spread_bps,
            estimated_fee_bps=fee_bps,
            estimated_slippage_bps=slippage_bps,
            edge_floor_bps=edge_floor_bps,
        )

    async def get_funding(self, symbol: str) -> Optional[Funding]:
        norm_symbol = self.normalize_symbol(symbol)
        try:
            resp = await self._request("/fapi/v1/premiumIndex", params={"symbol": norm_symbol})
            data = resp.json()
            return Funding(
                mark_price=float(data.get("markPrice", 0.0)),
                funding_rate=float(data.get("lastFundingRate", 0.0)),
                next_funding_time_ms=int(data.get("nextFundingTime", 0)),
            )
        except Exception:
            logger.warning("Failed to fetch funding for {symbol}", symbol=norm_symbol)
            return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        norm_symbol = self.normalize_symbol(symbol)
        try:
            resp = await self._request("/fapi/v1/openInterest", params={"symbol": norm_symbol})
            data = resp.json()
            return float(data.get("openInterest", 0.0))
        except Exception:
            logger.warning("Failed to fetch open interest for {symbol}", symbol=norm_symbol)
            return None

    async def get_structural_blocks(self, symbol: str, include_hourly: bool = True) -> Dict[str, object]:
        one_m = await self.get_candles(symbol, "1m", limit=self.config.expected_windows["1m"])
        fifteen_m = await self.get_candles(symbol, "15m", limit=self.config.expected_windows["15m"])
        hourly = None
        if include_hourly:
            hourly = await self.get_candles(symbol, "1h", limit=self.config.expected_windows["1h"])
        micro = await self.get_micro(symbol)
        funding = await self.get_funding(symbol)
        oi = await self.get_open_interest(symbol)
        return {
            "1m": one_m,
            "15m": fifteen_m,
            "1h": hourly,
            "micro": micro,
            "funding": funding,
            "open_interest": oi,
        }


__all__ = [
    "BinanceMarketData",
    "Candle",
    "MarketMicro",
    "Funding",
    "IntervalBlock",
    "MarketDataConfig",
]

"""Feature pipeline abstractions for the strategy agent.

This module encapsulates the data-fetch and feature-computation steps used by
strategy runtimes. Introducing a dedicated pipeline object means the decision
coordinator no longer needs direct access to the market data source or feature
computer—everything is orchestrated by the pipeline.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from loguru import logger

from valuecell.agents.common.trading.models import (
    Candle,
    CandleConfig,
    FeaturesPipelineResult,
    FeatureVector,
    UserRequest,
)

from ..data.interfaces import BaseMarketDataSource
from ..data.market import SimpleMarketDataSource
from .candle import SimpleCandleFeatureComputer
from .interfaces import (
    BaseFeaturesPipeline,
    CandleBasedFeatureComputer,
)
from .market_snapshot import MarketSnapshotFeatureComputer


_INTERVAL_MULTIPLIERS: Dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
}


def _interval_to_seconds(interval: str) -> int:
    unit = interval[-1]
    multiplier = _INTERVAL_MULTIPLIERS.get(unit)
    if multiplier is None:
        raise ValueError(f"Unsupported interval unit: {interval}")
    return int(interval[:-1]) * multiplier


def _resample_candles(
    candles: Iterable[Candle],
    *,
    target_interval: str,
    target_interval_ms: int,
    target_lookback: int | None = None,
) -> List[Candle]:
    grouped: Dict[str, List[Candle]] = defaultdict(list)
    for candle in candles:
        grouped[candle.instrument.symbol].append(candle)

    resampled: List[Candle] = []
    for symbol, series in grouped.items():
        series.sort(key=lambda item: item.ts)
        buckets: Dict[int, Dict[str, object]] = {}
        for candle in series:
            bucket_start = (candle.ts // target_interval_ms) * target_interval_ms
            bucket = buckets.get(bucket_start)
            if bucket is None:
                bucket = {
                    "instrument": candle.instrument,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "ts": bucket_start + target_interval_ms,
                }
                buckets[bucket_start] = bucket
            else:
                bucket["close"] = candle.close
                bucket["high"] = max(bucket["high"], candle.high)
                bucket["low"] = min(bucket["low"], candle.low)
                bucket["volume"] = float(bucket["volume"]) + float(candle.volume)

        bucket_starts = sorted(buckets.keys())
        symbol_resampled: List[Candle] = []
        for start in bucket_starts:
            bucket = buckets[start]
            symbol_resampled.append(
                Candle(
                    ts=int(bucket["ts"]),
                    instrument=bucket["instrument"],
                    open=float(bucket["open"]),
                    high=float(bucket["high"]),
                    low=float(bucket["low"]),
                    close=float(bucket["close"]),
                    volume=float(bucket["volume"]),
                    interval=target_interval,
                )
            )

        if target_lookback is not None and len(symbol_resampled) > target_lookback:
            symbol_resampled = symbol_resampled[-target_lookback:]

        resampled.extend(symbol_resampled)

    resampled.sort(key=lambda item: item.ts)
    return resampled


class DefaultFeaturesPipeline(BaseFeaturesPipeline):
    """Default pipeline using the simple data source and feature computer."""

    def __init__(
        self,
        *,
        request: UserRequest,
        market_data_source: BaseMarketDataSource,
        candle_feature_computer: CandleBasedFeatureComputer,
        market_snapshot_computer: MarketSnapshotFeatureComputer,
        candle_configurations: Optional[List[CandleConfig]] = None,
    ) -> None:
        self._request = request
        self._market_data_source = market_data_source
        self._candle_feature_computer = candle_feature_computer
        self._symbols = list(dict.fromkeys(request.trading_config.symbols))
        self._market_snapshot_computer = market_snapshot_computer
        self._candle_configurations = candle_configurations
        self._candle_configurations = candle_configurations or self._build_default_candle_configs()

    async def build(self) -> FeaturesPipelineResult:
        """
        Fetch candles and market snapshot, compute feature vectors concurrently,
        and combine results.
        """

        async def _fetch_candles(interval: str, lookback: int) -> List[FeatureVector]:
            """Fetches candles and computes features for a single (interval, lookback) pair."""
            _candles = await self._market_data_source.get_recent_candles(
                self._symbols, interval, lookback
            )
            if not _candles and interval in {"15m", "1h"}:
                logger.warning(
                    "No candles returned for interval={}. Falling back to 1m resampling.",
                    interval,
                )
                _candles = await self._resample_from_minute(interval, lookback)

            if not _candles:
                return []

            return self._candle_feature_computer.compute_features(candles=_candles)

        async def _fetch_market_features() -> List[FeatureVector]:
            """Fetches market snapshot for all symbols and computes features."""
            market_snapshot = await self._market_data_source.get_market_snapshot(
                self._symbols
            )
            market_snapshot = market_snapshot or {}
            return self._market_snapshot_computer.build(
                market_snapshot, self._request.exchange_config.exchange_id
            )

        logger.info(
            f"Starting concurrent data fetching for {len(self._candle_configurations)} candle sets and markets snapshot..."
        )
        tasks = [
            _fetch_candles(config.interval, config.lookback)
            for config in self._candle_configurations
        ]
        tasks.append(_fetch_market_features())

        # results = [ [candle_features_1], [candle_features_2], ..., [market_features] ]
        results = await asyncio.gather(*tasks)
        logger.info("Concurrent data fetching complete.")

        market_features: List[FeatureVector] = results.pop()

        # Flatten the list of lists of candle features
        candle_features: List[FeatureVector] = list(
            itertools.chain.from_iterable(results)
        )

        candle_features.extend(market_features)

        return FeaturesPipelineResult(features=candle_features)

    @classmethod
    def from_request(cls, request: UserRequest) -> DefaultFeaturesPipeline:
        """Factory creating the default pipeline from a user request."""
        market_data_source = SimpleMarketDataSource(
            exchange_id=request.exchange_config.exchange_id
        )
        candle_feature_computer = SimpleCandleFeatureComputer()
        market_snapshot_computer = MarketSnapshotFeatureComputer()
        return cls(
            request=request,
            market_data_source=market_data_source,
            candle_feature_computer=candle_feature_computer,
            market_snapshot_computer=market_snapshot_computer,
        )

    async def _resample_from_minute(
        self, target_interval: str, target_lookback: int
    ) -> List[Candle]:
        target_seconds = _interval_to_seconds(target_interval)
        minute_seconds = _interval_to_seconds("1m")
        if target_seconds % minute_seconds != 0:
            logger.warning(
                "Cannot resample 1m candles into unsupported interval={}", target_interval
            )
            return []

        ratio = target_seconds // minute_seconds
        # Ensure we fetch enough 1m bars to build the requested target lookback plus a buffer
        minute_lookback = (ratio * target_lookback) + 5
        minute_candles = await self._market_data_source.get_recent_candles(
            self._symbols, "1m", minute_lookback
        )
        if not minute_candles:
            logger.warning(
                "Unable to fetch 1m candles for resampling to interval={}, lookback={}",
                target_interval,
                target_lookback,
            )
            return []

        target_interval_ms = target_seconds * 1000
        resampled = _resample_candles(
            minute_candles,
            target_interval=target_interval,
            target_interval_ms=target_interval_ms,
            target_lookback=target_lookback,
        )
        logger.info(
            "Resampled {} 1m candles into {} {} candles (requested lookback {}).",
            len(minute_candles),
            len(resampled),
            target_interval,
            target_lookback,
        )
        return resampled

    def _build_default_candle_configs(self) -> list[CandleConfig]:
        """Return default candle intervals, adapting to exchange support.

        OKX does not offer 1-second OHLC candles, so it skips 1s but still
        requests 1m/15m/1h intervals. Other exchanges keep the 1s feed while
        also fetching the higher intervals to support multi-timeframe signals.
        """

        if self._request.exchange_config.exchange_id.lower() == "okx":
            return [
                CandleConfig(interval="1m", lookback=60 * 12),
                CandleConfig(interval="15m", lookback=96),
                CandleConfig(interval="1h", lookback=72),
            ]

        return [
            CandleConfig(interval="1s", lookback=60 * 3),
            CandleConfig(interval="1m", lookback=60 * 12),
            CandleConfig(interval="15m", lookback=96),
            CandleConfig(interval="1h", lookback=72),
        ]

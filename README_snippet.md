# Binance Market Data Layer (USDⓈ-M Futures)

This module replaces the raw candle fetcher with a resilient, futures-only data layer for Binance USDⓈ-M perpetuals. It always prefers native FAPI endpoints, falls back to CCXT futures, and only resamples from 1m data when necessary.

## Fallback order
1. **FAPI direct** (`/fapi/v1/klines` for 1m/15m/1h).
2. **CCXT futures** (`defaultType=future`, market type validated as USDⓈ-M).
3. **Resample** from 1m candles for 15m/1h if both layers fail. 1m itself will never resample and is marked `missing` when unavailable.

Each `IntervalBlock` reports `source` (`fapi`/`ccxt`/`resampled`/`missing`), `missing` flag, and coverage ratio. Failures are logged with fallback reasons and counters to avoid thundering herds.

## Coverage & guardrails
- Resampling keeps the first open, last close, high/low extrema, and summed volume.
- Buckets require ≥85% of expected 1m candles (15 for 15m, 60 for 1h) or they are dropped. Low coverage or insufficient history marks the block as `missing` to prevent indicator `None` propagation.

## Microstructure and costs
`get_structural_blocks(symbol)` also returns:
- Best bid/ask, spread (bps), estimated taker fees, slippage floor, and `edge_floor_bps` = `(2*fee + spread + slippage) * edge_mult`.
- Funding (mark price, rate, next funding time) and open interest.

## Rate limiting and cache
- FAPI calls use `EndpointRateLimiter` to respect local quotas and back off on 429/5xx.
- `/exchangeInfo` is cached for 1h with manual `clear_exchangeinfo_cache` support.

## Testing
Run `uv run pytest` to exercise normalization, parsing, fallback to CCXT with market validation, resampling/coverage, cache TTL, and cost floor math.

from valuecell.agents.common.trading.features.pipeline import _interval_to_seconds, _resample_candles
from valuecell.agents.common.trading.models import Candle, InstrumentRef


def _make_minute_candle(index: int) -> Candle:
    base_ts = 0
    ts = base_ts + (index * 60_000)
    price = float(index)
    return Candle(
        ts=ts,
        instrument=InstrumentRef(symbol="BTCUSDT", exchange_id="okx"),
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price + 0.1,
        volume=1.0,
        interval="1m",
    )


def _make_minute_candle_for_symbol(
    index: int, *, symbol: str, price_offset: float = 0.0
) -> Candle:
    base_ts = 0
    ts = base_ts + (index * 60_000)
    price = float(index) + price_offset
    return Candle(
        ts=ts,
        instrument=InstrumentRef(symbol=symbol, exchange_id="okx"),
        open=price,
        high=price + 0.5,
        low=price - 0.5,
        close=price + 0.1,
        volume=1.0,
        interval="1m",
    )


def test_resample_minutes_to_quarter_hour():
    candles = [_make_minute_candle(i) for i in range(30)]
    target_interval = "15m"
    interval_ms = _interval_to_seconds(target_interval) * 1000

    resampled = _resample_candles(
        candles,
        target_interval=target_interval,
        target_interval_ms=interval_ms,
        target_lookback=2,
    )

    assert len(resampled) == 2

    first, second = resampled
    assert first.open == candles[0].open
    assert first.close == candles[14].close
    assert first.high == candles[14].high
    assert first.low == candles[0].low
    assert first.volume == 15.0

    assert second.open == candles[15].open
    assert second.close == candles[29].close
    assert second.volume == 15.0


def test_resample_minutes_to_hour_trims_lookback():
    candles = [_make_minute_candle(i) for i in range(120)]
    target_interval = "1h"
    interval_ms = _interval_to_seconds(target_interval) * 1000

    resampled = _resample_candles(
        candles,
        target_interval=target_interval,
        target_interval_ms=interval_ms,
        target_lookback=1,
    )

    assert len(resampled) == 1
    last = resampled[-1]
    assert last.open == candles[60].open
    assert last.close == candles[119].close
    assert last.volume == 60.0


def test_resample_preserves_lookback_per_symbol():
    candles: list[Candle] = []
    for index in range(30):
        candles.append(
            _make_minute_candle_for_symbol(
                index, symbol="BTCUSDT", price_offset=0.0
            )
        )
        candles.append(
            _make_minute_candle_for_symbol(
                index, symbol="ETHUSDT", price_offset=100.0
            )
        )

    target_interval = "15m"
    interval_ms = _interval_to_seconds(target_interval) * 1000

    resampled = _resample_candles(
        candles,
        target_interval=target_interval,
        target_interval_ms=interval_ms,
        target_lookback=1,
    )

    assert len(resampled) == 2
    by_symbol = {candle.instrument.symbol: candle for candle in resampled}

    btc = by_symbol["BTCUSDT"]
    assert btc.open == candles[30].open
    assert btc.close == candles[58].close

    eth = by_symbol["ETHUSDT"]
    assert eth.open == candles[31].open
    assert eth.close == candles[59].close

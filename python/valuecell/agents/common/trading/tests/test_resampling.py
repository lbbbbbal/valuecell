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

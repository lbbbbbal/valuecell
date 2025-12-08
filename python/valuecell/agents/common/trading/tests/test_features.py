from valuecell.agents.common.trading.models import FeatureVector, InstrumentRef
from valuecell.agents.common.trading.utils import group_features


def _make_feature(symbol: str, interval: str, group_by_key: str | None = None):
    meta = {"interval": interval}
    if group_by_key:
        meta["group_by_key"] = group_by_key

    return FeatureVector(
        ts=1,
        instrument=InstrumentRef(symbol=symbol),
        values={"price.close": 1.0},
        meta=meta,
    )


def test_group_features_falls_back_to_interval_keys():
    features = [
        _make_feature("BTCUSDT", "1m"),
        _make_feature("BTCUSDT", "15m"),
        _make_feature("BTCUSDT", "1h"),
    ]

    grouped = group_features(features)

    assert set(grouped.keys()) == {"interval_1m", "interval_15m", "interval_1h"}
    for key, payloads in grouped.items():
        assert payloads[0]["meta"]["interval"] == key.replace("interval_", "")


def test_group_features_prefers_explicit_group_key():
    features = [_make_feature("BTCUSDT", "1m", group_by_key="market_snapshot")]

    grouped = group_features(features)

    assert set(grouped.keys()) == {"market_snapshot"}
    assert grouped["market_snapshot"][0]["meta"]["interval"] == "1m"

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
    FEATURE_GROUP_BY_SENTIMENT,
)
from valuecell.agents.common.trading.decision.prompt_based.composer import (
    compute_hybrid_scoreboard,
)
from valuecell.agents.common.trading.models import (
    ComposeContext,
    FeatureVector,
    InstrumentRef,
    PortfolioView,
    TradeDigest,
)


def _base_context(features):
    return ComposeContext(
        ts=0,
        compose_id="cmp-1",
        features=features,
        portfolio=PortfolioView(ts=0, account_balance=1000.0, positions={}),
        digest=TradeDigest(ts=0, by_instrument={}),
    )


def _market_snapshot(symbol: str, change_pct: float) -> FeatureVector:
    return FeatureVector(
        ts=0,
        instrument=InstrumentRef(symbol=symbol, exchange_id="binance"),
        values={"price.change_pct": change_pct, "price.volume": 1000.0},
        meta={FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_MARKET_SNAPSHOT},
    )


def _sentiment_feature(symbol: str, score: float) -> FeatureVector:
    return FeatureVector(
        ts=0,
        instrument=InstrumentRef(symbol=symbol, exchange_id="binance"),
        values={"sentiment.score": score},
        meta={FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_SENTIMENT},
    )


def test_hybrid_weights_shift_on_extreme_sentiment():
    features = [
        _market_snapshot("BTC-USDT", change_pct=2.0),
        _sentiment_feature("BTC-USDT", score=9.2),
    ]

    board = compute_hybrid_scoreboard(_base_context(features))

    entry = board["BTC-USDT"]
    assert entry["weights"]["sentiment"] == 0.6
    assert entry["weights"]["technical"] == 0.4
    assert entry["final_score"] >= entry["technical_score"]


def test_hybrid_weights_default_to_technical_bias():
    features = [
        _market_snapshot("ETH-USDT", change_pct=-1.0),
        _sentiment_feature("ETH-USDT", score=6.0),
    ]

    board = compute_hybrid_scoreboard(_base_context(features))

    entry = board["ETH-USDT"]
    assert entry["weights"]["sentiment"] == 0.4
    assert entry["weights"]["technical"] == 0.6
    assert entry["final_score"] < entry["sentiment_score"]


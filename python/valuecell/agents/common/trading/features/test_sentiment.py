import pytest

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_SENTIMENT,
)
from valuecell.agents.common.trading.features.sentiment import (
    SentimentFeatureComputer,
)
from valuecell.agents.sentiment_agent.models import SentimentPulse


@pytest.mark.asyncio
async def test_sentiment_feature_computer_builds_vectors():
    async def _fake_fetcher(symbols):
        return [
            SentimentPulse(
                symbol=symbols[0],
                score=8.5,
                sample_size=12,
                highlights="whale inflows",
                sources=["cryptopanic"],
            )
        ]

    computer = SentimentFeatureComputer(fetcher=_fake_fetcher)

    vectors = await computer.compute_features(["SOL-USDT"], exchange_id="okx")

    assert len(vectors) == 1
    fv = vectors[0]
    assert fv.values["sentiment.score"] == 8.5
    assert fv.meta[FEATURE_GROUP_BY_KEY] == FEATURE_GROUP_BY_SENTIMENT
    assert fv.instrument.exchange_id == "okx"


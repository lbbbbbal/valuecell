"""Sentiment feature builder for strategy agents.

This module integrates lightweight social-sentiment pulses so the
composer can combine technical and narrative momentum signals. It is
designed to be resilient when third-party APIs or credentials are
unavailable: missing data simply yields no sentiment features and the
hybrid scorer will fall back to technical signals.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Iterable, List

from loguru import logger

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_SENTIMENT,
)
from valuecell.agents.common.trading.models import FeatureVector, InstrumentRef
from valuecell.agents.sentiment_agent.models import SentimentPulse
from valuecell.agents.sentiment_agent.sources import collect_sentiment_pulses


class SentimentFeatureComputer:
    """Wraps social-sentiment fetchers into FeatureVector outputs."""

    def __init__(
        self,
        fetcher: Callable[[List[str]], Awaitable[Iterable[SentimentPulse]]]
        | None = None,
    ) -> None:
        self._fetcher = fetcher or collect_sentiment_pulses

    async def compute_features(
        self, symbols: List[str], exchange_id: str | None
    ) -> List[FeatureVector]:
        """Return sentiment FeatureVectors for the requested symbols.

        Missing fetchers or upstream failures are logged and result in an
        empty feature list so the decision loop can continue without
        blocking on social-data availability.
        """

        try:
            pulses = await self._fetcher(symbols)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment fetcher failed: {}", exc)
            return []

        now_ts = int(time.time() * 1000)
        features: List[FeatureVector] = []
        for pulse in pulses:
            features.append(
                FeatureVector(
                    ts=int(pulse.ts or now_ts),
                    instrument=InstrumentRef(
                        symbol=pulse.symbol, exchange_id=exchange_id
                    ),
                    values={
                        "sentiment.score": float(pulse.score),
                        "sentiment.sample_size": pulse.sample_size or 0,
                        "sentiment.sources": ",".join(pulse.sources or []),
                        "sentiment.highlights": (pulse.highlights or None),
                    },
                    meta={FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_SENTIMENT},
                )
            )

        return features


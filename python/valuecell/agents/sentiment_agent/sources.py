from __future__ import annotations

import os
import time
from typing import Iterable, List

import httpx
from loguru import logger

from .models import SentimentPulse


def _clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


async def _fetch_cryptopanic(symbol: str, token: str, client: httpx.AsyncClient) -> float:
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": token, "currencies": symbol, "filter": "rising"}
    resp = await client.get(url, params=params, timeout=10.0)
    resp.raise_for_status()
    payload = resp.json()
    posts = payload.get("results") or []
    if not posts:
        return 5.0

    positive = 0
    negative = 0
    important = 0
    for post in posts:
        votes = post.get("votes") or {}
        positive += int(votes.get("positive") or 0)
        negative += int(votes.get("negative") or 0)
        important += int(votes.get("important") or 0)

    baseline = 5.0 + (important * 0.05)
    if positive + negative == 0:
        return _clamp(baseline)

    ratio = (positive - negative) / max(1, positive + negative)
    return _clamp(baseline + ratio * 5.0)


async def _fetch_lunarcrush(symbol: str, api_key: str, client: httpx.AsyncClient) -> float:
    url = "https://lunarcrush.com/api4/public/coins"
    params = {"symbols": symbol, "data_points": 0, "interval": "day"}
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = await client.get(url, params=params, headers=headers, timeout=10.0)
    resp.raise_for_status()
    payload = resp.json() or {}
    data = (payload.get("data") or [])
    if not data:
        return 5.0

    coin = data[0]
    galaxy_score = coin.get("galaxy_score")
    if galaxy_score is None:
        return 5.0
    return _clamp(float(galaxy_score) / 10.0)


async def collect_sentiment_pulses(symbols: List[str]) -> Iterable[SentimentPulse]:
    """Aggregate social sentiment across configured providers.

    Providers are optional: if an API key/token is missing the provider is
    skipped. A neutral score (5.0) is used when no signals are returned for
    a symbol so downstream consumers always receive a bounded value.
    """

    cryptopanic_token = os.getenv("CRYPTOPANIC_TOKEN")
    lunarcrush_key = os.getenv("LUNARCRUSH_API_KEY")

    if not cryptopanic_token and not lunarcrush_key:
        logger.info("No sentiment providers configured; skipping social sentiment fetch")
        return []

    pulses: List[SentimentPulse] = []
    ts_ms = int(time.time() * 1000)

    async with httpx.AsyncClient() as client:
        for symbol in symbols:
            scores: List[float] = []
            sources: List[str] = []

            if cryptopanic_token:
                try:
                    scores.append(
                        await _fetch_cryptopanic(symbol, cryptopanic_token, client)
                    )
                    sources.append("cryptopanic")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cryptopanic sentiment fetch failed for {}: {}", symbol, exc)

            if lunarcrush_key:
                try:
                    scores.append(
                        await _fetch_lunarcrush(symbol, lunarcrush_key, client)
                    )
                    sources.append("lunarcrush")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("LunarCrush sentiment fetch failed for {}: {}", symbol, exc)

            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            pulses.append(
                SentimentPulse(
                    symbol=symbol,
                    score=_clamp(avg_score),
                    sample_size=None,
                    highlights=None,
                    sources=sources,
                    ts=ts_ms,
                )
            )

    return pulses


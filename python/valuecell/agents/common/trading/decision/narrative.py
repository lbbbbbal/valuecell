from __future__ import annotations

"""Lightweight narrative aggregation for news + social sentiment.

This module keeps news collection (NewsAgent) separate from crowd
sentiment (SentimentAgent) and fuses them into a single narrative
momentum signal that StrategyAgent can consume without duplicating
responsibilities. The output keeps both sub-scores plus agreement
flags so downstream logic can trace how the blended score was formed.
"""

from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_AGREEMENT_BOOST: float = 0.5
DEFAULT_TECHNICAL_FLOOR: float = 3.0


class NewsSignal(BaseModel):
    """Structured news sentiment produced by NewsAgent."""

    news_score: float = Field(..., ge=0.0, le=10.0)
    direction: str = Field(..., description="bullish | bearish | neutral")
    rationale: Optional[str] = Field(
        default=None,
        description="Optional explanation of the news-derived score",
    )


class SentimentSignal(BaseModel):
    """Structured social sentiment produced by SentimentAgent."""

    social_score: float = Field(..., ge=0.0, le=10.0)
    direction: str = Field(..., description="bullish | bearish | neutral")
    sentiment_score: Optional[float] = Field(
        default=None,
        description=(
            "Raw social sentiment score to surface to automated trading prompts"
        ),
    )
    rationale: Optional[str] = Field(
        default=None, description="Optional explanation of the social score"
    )


class NarrativeSignal(BaseModel):
    """Fused narrative signal used by StrategyAgent."""

    narrative_score: float = Field(..., ge=0.0, le=10.0)
    news_score: float = Field(..., ge=0.0, le=10.0)
    social_score: float = Field(..., ge=0.0, le=10.0)
    agreement_flag: bool = Field(
        ..., description="True when news and social directions align"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Merged rationale with provenance hints",
    )


class SignalMix(BaseModel):
    """Final blended score and gating flags consumed by StrategyAgent."""

    final_score: float = Field(..., ge=0.0, le=10.0)
    narrative_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    technical_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    narrative_weight: float = Field(..., ge=0.0, le=1.0)
    technical_weight: float = Field(..., ge=0.0, le=1.0)
    agreement_flag: bool = Field(...)
    micro_probe_only: bool = Field(
        ..., description="Guardrail: True when technical floor is breached"
    )
    mode: str = Field(
        ..., description="technical_only | blended | agreement_tilt"
    )
    technical_floor: float = Field(..., ge=0.0, description="Safety valve floor")


def _directions_align(news_direction: str, social_direction: str) -> bool:
    news_dir = (news_direction or "").strip().lower()
    social_dir = (social_direction or "").strip().lower()
    if not news_dir or not social_dir:
        return False
    if news_dir == "neutral" or social_dir == "neutral":
        return False
    return news_dir == social_dir


def build_narrative_signal(
    news_signal: Optional[NewsSignal],
    sentiment_signal: Optional[SentimentSignal],
    *,
    agreement_boost: float = DEFAULT_AGREEMENT_BOOST,
) -> Optional[NarrativeSignal]:
    """Fuse news + social sentiment into a NarrativeSignal.

    MVP rule-set:
    - Base narrative score is the average of news_score and social_score.
    - If directions agree AND either input score > 8, apply a small boost
      (capped at 10).
    """

    if news_signal is None or sentiment_signal is None:
        return None

    agreement_flag = _directions_align(
        news_signal.direction, sentiment_signal.direction
    )
    base_score = 0.5 * news_signal.news_score + 0.5 * sentiment_signal.social_score

    if agreement_flag and (
        news_signal.news_score > 8 or sentiment_signal.social_score > 8
    ):
        # Apply a bounded boost with headroom buffer to avoid sudden jumps at the cap
        headroom = max(0.0, 10.0 - base_score)
        base_score += min(headroom, max(0.0, agreement_boost))

    base_score = min(10.0, max(0.0, base_score))

    rationale = news_signal.rationale or sentiment_signal.rationale
    if news_signal.rationale and sentiment_signal.rationale:
        rationale = (
            f"News: {news_signal.rationale}\nSocial: {sentiment_signal.rationale}"
        )

    return NarrativeSignal(
        narrative_score=base_score,
        news_score=news_signal.news_score,
        social_score=sentiment_signal.social_score,
        agreement_flag=agreement_flag,
        rationale=rationale,
    )


def mix_signals(
    *,
    technical_score: float | None,
    narrative_signal: Optional[NarrativeSignal],
    technical_floor: float = DEFAULT_TECHNICAL_FLOOR,
) -> SignalMix:
    """Blend narrative and technical scores with a safety floor.

    - Default weights: 0.4 narrative / 0.6 technical.
    - If narrative_score > 8 AND agreement_flag is true, tilt to
      0.6 narrative / 0.4 technical.
    - If narrative is missing, fall back to 100% technical.
    - If technical falls below the floor, mark micro_probe_only=True to
      prevent primary positions regardless of narrative strength. This is the
      single truth source for the micro-probe gating flag.
    """

    technical_value = max(0.0, float(technical_score or 0.0))
    micro_probe_only = technical_value < technical_floor

    if narrative_signal is None:
        return SignalMix(
            final_score=min(10.0, technical_value),
            narrative_score=None,
            technical_score=technical_value,
            narrative_weight=0.0,
            technical_weight=1.0,
            agreement_flag=False,
            micro_probe_only=micro_probe_only,
            mode="technical_only",
            technical_floor=technical_floor,
        )

    narrative_weight = 0.4
    technical_weight = 0.6
    mode = "blended"

    if narrative_signal.narrative_score > 8.0 and narrative_signal.agreement_flag:
        narrative_weight, technical_weight = 0.6, 0.4
        mode = "agreement_tilt"

    final_score = (
        narrative_weight * narrative_signal.narrative_score
        + technical_weight * technical_value
    )

    return SignalMix(
        final_score=min(10.0, final_score),
        narrative_score=narrative_signal.narrative_score,
        technical_score=technical_value,
        narrative_weight=narrative_weight,
        technical_weight=technical_weight,
        agreement_flag=narrative_signal.agreement_flag,
        micro_probe_only=micro_probe_only,
        mode=mode,
        technical_floor=technical_floor,
    )

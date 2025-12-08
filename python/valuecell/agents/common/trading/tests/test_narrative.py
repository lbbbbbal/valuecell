from valuecell.agents.common.trading.decision.narrative import (
    DEFAULT_TECHNICAL_FLOOR,
    NewsSignal,
    SentimentSignal,
    build_narrative_signal,
    mix_signals,
)


def test_build_narrative_with_agreement_boost() -> None:
    news = NewsSignal(news_score=9.0, direction="bullish", rationale="regulatory win")
    social = SentimentSignal(
        social_score=8.0,
        direction="bullish",
        sentiment_score=0.82,
        rationale="community trending up",
    )

    narrative = build_narrative_signal(news, social)

    assert narrative is not None
    assert narrative.agreement_flag is True
    # Base average 8.5, boosted to 9.0 with agreement bump capped at 10
    assert narrative.narrative_score == 9.0
    assert narrative.news_score == news.news_score
    assert narrative.social_score == social.social_score


def test_build_narrative_without_agreement() -> None:
    news = NewsSignal(news_score=9.0, direction="bullish")
    social = SentimentSignal(social_score=4.0, direction="bearish")

    narrative = build_narrative_signal(news, social)

    assert narrative is not None
    assert narrative.agreement_flag is False
    assert narrative.narrative_score == 6.5


def test_mix_signals_switches_weights_on_strong_alignment() -> None:
    narrative = build_narrative_signal(
        NewsSignal(news_score=9.0, direction="bullish"),
        SentimentSignal(social_score=9.0, direction="bullish"),
    )
    result = mix_signals(technical_score=6.0, narrative_signal=narrative)

    assert result.mode == "agreement_tilt"
    assert result.narrative_weight == 0.6
    assert result.technical_weight == 0.4
    # Boosted narrative (9.5) tilts the blend upward
    assert round(result.final_score, 1) == 8.1


def test_mix_signals_falls_back_to_technical_only_when_missing_narrative() -> None:
    result = mix_signals(technical_score=7.5, narrative_signal=None)

    assert result.mode == "technical_only"
    assert result.narrative_weight == 0.0
    assert result.technical_weight == 1.0
    assert result.final_score == 7.5


def test_micro_probe_flag_when_technical_below_floor() -> None:
    narrative = build_narrative_signal(
        NewsSignal(news_score=7.0, direction="bullish"),
        SentimentSignal(social_score=7.0, direction="bullish"),
    )
    result = mix_signals(
        technical_score=2.0,
        narrative_signal=narrative,
        technical_floor=DEFAULT_TECHNICAL_FLOOR,
    )

    assert result.micro_probe_only is True
    assert result.final_score == 4.0


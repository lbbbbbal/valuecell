from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SentimentPulse(BaseModel):
    """Aggregated social sentiment for a symbol."""

    symbol: str = Field(..., description="Trading symbol, e.g., BTC-USDT")
    score: float = Field(
        ..., ge=0.0, le=10.0, description="Composite score in [0,10] where 10 is max hype"
    )
    sample_size: Optional[int] = Field(
        default=None, description="Number of raw items counted for this pulse"
    )
    highlights: Optional[str] = Field(
        default=None, description="Concise highlight text summarizing catalysts"
    )
    sources: List[str] = Field(
        default_factory=list, description="Data providers used to derive the score"
    )
    ts: Optional[int] = Field(
        default=None, description="Timestamp (ms) for the pulse measurement"
    )


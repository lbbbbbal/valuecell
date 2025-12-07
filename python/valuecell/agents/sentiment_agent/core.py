from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

from valuecell.agents.sentiment_agent.prompts import SENTIMENT_AGENT_INSTRUCTIONS
from valuecell.agents.sentiment_agent.sources import collect_sentiment_pulses
from valuecell.adapters.models import create_model_for_agent
from valuecell.core.agent.responses import streaming
from valuecell.core.types import BaseAgent, StreamResponse


def _parse_symbols(payload: str) -> List[str]:
    try:
        data = json.loads(payload)
        symbols = data.get("symbols") if isinstance(data, dict) else None
        if isinstance(symbols, list):
            return [str(sym).upper() for sym in symbols if str(sym).strip()]
    except Exception:  # noqa: BLE001
        pass

    parts = re.split(r"[\s,]+", payload)
    return [p.upper() for p in parts if p]


class SentimentAgent(BaseAgent):
    """Agent that aggregates social sentiment for requested symbols."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = create_model_for_agent("sentiment_agent")

    async def stream(
        self,
        query: str,
        conversation_id: str,
        task_id: str,
        dependencies: Optional[Dict] = None,
    ) -> AsyncGenerator[StreamResponse, None]:
        symbols = _parse_symbols(query)
        if not symbols:
            yield streaming.message_chunk("Please provide one or more symbols.")
            yield streaming.done()
            return

        logger.info("Fetching sentiment for %s", symbols)
        pulses = await collect_sentiment_pulses(symbols)

        if not pulses:
            yield streaming.message_chunk(
                "No social-sentiment providers configured or no data returned."
            )
            yield streaming.done()
            return

        lines = ["Social sentiment snapshot:"]
        for pulse in pulses:
            alert = " 🚀" if pulse.score >= 8 else ""
            lines.append(
                f"- {pulse.symbol}: {pulse.score:.1f}/10{alert}"
                f" (sources: {', '.join(pulse.sources) or 'n/a'})"
            )
            if pulse.highlights:
                lines.append(f"  drivers: {pulse.highlights}")

        yield streaming.message_chunk("\n".join(lines))
        yield streaming.done()

    async def run(self, query: str, **kwargs: Any) -> str:
        symbols = _parse_symbols(query)
        pulses = await collect_sentiment_pulses(symbols)
        if not pulses:
            return "No social-sentiment data available. Configure a provider to enable it."

        parts = []
        for pulse in pulses:
            parts.append(
                f"{pulse.symbol}: {pulse.score:.1f}/10"
                f" (sources: {', '.join(pulse.sources) or 'n/a'})"
            )
            if pulse.highlights:
                parts.append(f"drivers: {pulse.highlights}")
        return " | ".join(parts)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Sentiment Agent",
            "description": "Aggregates social sentiment (CryptoPanic/LunarCrush) into 0-10 scores",
            "instructions": SENTIMENT_AGENT_INSTRUCTIONS,
        }


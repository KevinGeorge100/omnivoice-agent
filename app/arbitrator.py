"""Multi-engine turn boundary arbitration and Indic filler hold guard for OmniVoice."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Indic (Malayalam/Hinglish) and English trailing thought-fillers / hesitations
TRAILING_THOUGHT_FILLERS = {
    "aa",
    "aaa",
    "um",
    "umm",
    "uh",
    "uhh",
    "appol",
    "appo",
    "pinne",
    "ennu",
    "ennum",
    "ready",
    "alle",
    "kettuo",
    "nokku",
    "aaan",
    "aanu",
}


def should_hold_execution(transcript: str) -> bool:
    """Return True if the transcript ends with common English or Malayalam trailing fillers."""
    normalized = " ".join(str(transcript or "").strip().casefold().split())
    if not normalized:
        return False

    words = re.findall(r"[a-zà-öø-ÿ]+", normalized)
    if not words:
        return False

    last_word = words[-1]
    if last_word in TRAILING_THOUGHT_FILLERS:
        return True

    if normalized.endswith("...") or normalized.endswith("…"):
        return True

    return False


class TurnBoundaryArbiter:
    """Coordinates final transcript racing conditions between dual STT engines and guards turn boundaries."""

    def __init__(self, hold_delay_seconds: float = 0.25) -> None:
        self.hold_delay_seconds = hold_delay_seconds
        self.current_turn_id = 0
        self._claimed_turns: set[int] = set()
        self._lock = asyncio.Lock()
        self._last_processed_transcript = ""
        self._last_processed_at = 0.0

    async def process_final_transcript(
        self,
        transcript: str,
        source: str,
        on_accepted: Callable[..., Awaitable[None]],
        *,
        stt_latency_ms: float | None = None,
        turn_started_at: float | None = None,
    ) -> bool:
        """Arbitrate racing STT engines, claim the turn lock atomically, and execute valid turns."""
        cleaned = str(transcript or "").strip()
        if not cleaned:
            return False

        async with self._lock:
            now = time.monotonic()
            normalized = " ".join(cleaned.casefold().split())
            if (
                normalized == self._last_processed_transcript
                and (now - self._last_processed_at) < 1.5
            ):
                logger.info(
                    "Arbitrator: Discarded duplicate turn from %s: %r", source, cleaned
                )
                return False

            self.current_turn_id += 1
            target_turn_id = self.current_turn_id
            self._claimed_turns.add(target_turn_id)
            self._last_processed_transcript = normalized
            self._last_processed_at = now

        if should_hold_execution(cleaned):
            logger.info(
                "Arbitrator: Trailing filler detected in transcript %r; holding execution for %.0fms",
                cleaned,
                self.hold_delay_seconds * 1000,
            )
            await asyncio.sleep(self.hold_delay_seconds)

        await on_accepted(
            cleaned,
            source=source,
            stt_latency_ms=stt_latency_ms,
            turn_started_at=turn_started_at,
        )
        return True

    def reset(self) -> None:
        """Reset internal turn state."""
        self._claimed_turns.clear()
        self._last_processed_transcript = ""
        self._last_processed_at = 0.0

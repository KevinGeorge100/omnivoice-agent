"""Concurrency and transcript guards for one OmniVoice WebSocket session."""

import asyncio
import time


class TranscriptDeduplicator:
    """Reject immediate duplicate final transcripts from STT or browser retries."""

    def __init__(self, window_seconds: float = 1.5) -> None:
        self._window_seconds = window_seconds
        self._last_transcript = ""
        self._last_received_at = 0.0

    def accept(self, transcript: str) -> bool:
        normalized = " ".join(transcript.casefold().split())
        now = time.monotonic()
        is_duplicate = (
            normalized
            and normalized == self._last_transcript
            and now - self._last_received_at < self._window_seconds
        )
        self._last_transcript = normalized
        self._last_received_at = now
        return bool(normalized) and not is_duplicate


async def cancel_and_wait(tasks: set[asyncio.Task[None]]) -> None:
    """Cancel tracked tasks and await their cleanup without cancelling the caller."""
    current_task = asyncio.current_task()
    pending = [task for task in tasks if task is not current_task and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()

"""Concurrency and transcript guards for one OmniVoice WebSocket session."""

import asyncio
import re
import time


INCOMPLETE_TRAILING_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "been",
    "being",
    "but",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "to",
    "under",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
    "about",
    "after",
    "before",
    "because",
    "through",
    "until",
    "over",
    "around",
    "between",
    "within",
}


def should_process_transcript(transcript: str) -> bool:
    """Only accept a transcript when it looks like a complete utterance.

    This blocks common mid-sentence fragments such as "I want to" or "The flight from Kochi to"
    while still allowing short but valid phrases like "Hi" or "Okay".
    """
    normalized = " ".join(str(transcript or "").strip().split())
    if not normalized:
        return False

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", normalized)
    if not words:
        return False

    if len(words) == 1:
        return words[0].casefold() not in {"to", "from", "and", "or", "if", "when", "where", "why", "who", "which"}

    last_word = words[-1].casefold()
    if last_word in INCOMPLETE_TRAILING_WORDS:
        return False

    if last_word.endswith("ing") and len(words) <= 3:
        return False

    if normalized.endswith(("?", "!", ".")):
        return True

    if last_word in {"please", "kindly", "sir", "maam", "madam", "hello"}:
        return False

    return True


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


class SessionState:
    """Centralized atomic turn state and task tracking manager for one client session."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.response_tasks: set[asyncio.Task[None]] = set()
        self.transcription_tasks: set[asyncio.Task[None]] = set()
        self.stream_tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    def track_response(self, task: asyncio.Task[None]) -> None:
        self.response_tasks.add(task)
        task.add_done_callback(self.response_tasks.discard)

    def track_transcription(self, task: asyncio.Task[None]) -> None:
        self.transcription_tasks.add(task)
        task.add_done_callback(self.transcription_tasks.discard)

    def track_stream(self, task: asyncio.Task[None]) -> None:
        self.stream_tasks.add(task)
        task.add_done_callback(self.stream_tasks.discard)

    async def barge_in_atomic(self) -> None:
        """Atomically cancel all active response, transcription, and stream tasks."""
        async with self._lock:
            all_tasks = set()
            all_tasks.update(self.response_tasks)
            all_tasks.update(self.transcription_tasks)
            all_tasks.update(self.stream_tasks)

            current_task = asyncio.current_task()
            pending = [t for t in all_tasks if t is not current_task and not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            self.response_tasks.clear()
            self.transcription_tasks.clear()
            self.stream_tasks.clear()

    async def close_atomic(self) -> None:
        """Clean up all session tasks on WebSocket disconnect."""
        await self.barge_in_atomic()

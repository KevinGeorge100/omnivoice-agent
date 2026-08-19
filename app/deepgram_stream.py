"""Non-blocking Deepgram live transcription for browser audio streams."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from deepgram import AsyncDeepgramClient
from dotenv import load_dotenv

from app.config import is_configured


TranscriptHandler = Callable[[str, bool], Awaitable[None]]


class DeepgramLiveTranscriber:
    """Forward WebM audio frames to Deepgram and emit interim/final transcripts."""

    def __init__(self, on_transcript: TranscriptHandler) -> None:
        load_dotenv()
        self._on_transcript = on_transcript
        self._client: AsyncDeepgramClient | None = None
        self._connection: Any | None = None
        self._connection_context: Any | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._final_parts: list[str] = []

    @staticmethod
    def is_configured() -> bool:
        return is_configured("DEEPGRAM_API_KEY")

    async def start(self) -> None:
        """Open the Deepgram listen stream for containerized browser WebM audio."""
        if not self.is_configured():
            raise RuntimeError("DEEPGRAM_API_KEY is not configured")

        self._client = AsyncDeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
        self._connection_context = self._client.listen.v1.connect(
            model="nova-3",
            language="en-US",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            endpointing=300,
            vad_events=True,
        )
        self._connection = await self._connection_context.__aenter__()
        self._receive_task = asyncio.create_task(
            self._receive_transcripts(), name="deepgram-live-transcription"
        )

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Forward one incoming WebSocket binary frame to Deepgram."""
        if self._connection is not None:
            await self._connection.send_media(audio_chunk)

    async def _receive_transcripts(self) -> None:
        assert self._connection is not None
        while True:
            message = await self._connection.recv()
            if getattr(message, "type", None) != "Results":
                continue

            alternatives = getattr(getattr(message, "channel", None), "alternatives", [])
            transcript = alternatives[0].transcript.strip() if alternatives else ""
            if not transcript:
                continue

            is_final = bool(getattr(message, "is_final", False))
            speech_final = bool(getattr(message, "speech_final", False))

            if not is_final:
                await self._on_transcript(transcript, False)
                continue

            self._final_parts.append(transcript)
            if speech_final:
                completed_utterance = " ".join(self._final_parts).strip()
                self._final_parts.clear()
                if completed_utterance:
                    await self._on_transcript(completed_utterance, True)

    async def close(self) -> None:
        """Close the live stream and release the receiver task."""
        if self._receive_task is not None:
            self._receive_task.cancel()
            await asyncio.gather(self._receive_task, return_exceptions=True)
            self._receive_task = None

        if self._connection is not None:
            try:
                await self._connection.send_finalize()
                await self._connection.send_close_stream()
            finally:
                self._connection = None

        if self._connection_context is not None:
            await self._connection_context.__aexit__(None, None, None)
            self._connection_context = None

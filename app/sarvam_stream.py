"""Non-blocking Sarvam Saaras real-time transcription for 16 kHz PCM audio."""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import Awaitable, Callable
from typing import Any

from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI
from sarvamai.types.realtime_audio_input import RealtimeAudioInput

from app.config import is_configured


TranscriptHandler = Callable[[str, bool], Awaitable[None]]
VadHandler = Callable[[str], Awaitable[None]]
ErrorHandler = Callable[[str], Awaitable[None]]


class SarvamLiveTranscriber:
    """Bridge browser 16 kHz PCM frames to a persistent Saaras real-time WebSocket."""

    sample_rate = 16_000

    def __init__(
        self,
        on_transcript: TranscriptHandler,
        on_vad: VadHandler,
        on_error: ErrorHandler,
    ) -> None:
        load_dotenv()
        self._on_transcript = on_transcript
        self._on_vad = on_vad
        self._on_error = on_error
        self._client: AsyncSarvamAI | None = None
        self._connection_context: Any | None = None
        self._connection: Any | None = None
        self._receive_task: asyncio.Task[None] | None = None

    @staticmethod
    def is_configured() -> bool:
        return is_configured("SARVAM_API_KEY")

    async def start(self) -> None:
        """Open a single persistent real-time session for the caller's Malayalam turns."""
        if not self.is_configured():
            raise RuntimeError("SARVAM_API_KEY is not configured")

        self._client = AsyncSarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        self._connection_context = self._client.speech_to_text_realtime_streaming.connect(
            model="saaras:v3-realtime",
            mode="transcribe",
            language_code="ml-IN",
            sample_rate=str(self.sample_rate),
            encoding="linear16",
            endpointing="vad",
            stream_type="balanced",
        )
        self._connection = await self._connection_context.__aenter__()
        self._receive_task = asyncio.create_task(
            self._receive_messages(), name="sarvam-live-transcription"
        )

    async def send_pcm(self, audio_chunk: bytes) -> None:
        """Send one signed 16-bit little-endian mono PCM frame without blocking the loop."""
        if self._connection is None or not audio_chunk:
            return
        try:
            encoded_audio = base64.b64encode(audio_chunk).decode("ascii")
            await self._connection.send_realtime_audio_input(RealtimeAudioInput(audio=encoded_audio))
        except Exception:
            # Connection closed or socket write failed during streaming
            pass

    async def _receive_messages(self) -> None:
        assert self._connection is not None
        try:
            while True:
                message = await self._connection.recv()
                event_type = getattr(message, "event", None)
                if event_type == "vad.speech_start":
                    await self._on_vad("START_SPEECH")
                elif event_type == "vad.speech_end":
                    await self._on_vad("END_SPEECH")
                elif event_type in {"transcript.partial", "transcript.final"}:
                    transcript = str(getattr(message, "text", "")).strip()
                    if not transcript:
                        continue
                    await self._on_transcript(transcript, event_type == "transcript.final")
                elif event_type == "error":
                    await self._on_error(str(getattr(message, "message", "Sarvam streaming failed")))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._on_error("Sarvam streaming connection was interrupted")
            raise RuntimeError("Sarvam streaming connection failed") from error

    async def close(self) -> None:
        """Cancel the receive loop and close the streaming context exactly once."""
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._connection_context is not None:
            await self._connection_context.__aexit__(None, None, None)
            self._connection_context = None
            self._connection = None

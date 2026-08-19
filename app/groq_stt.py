"""Groq Whisper transcription for completed browser speech turns."""

import os

from dotenv import load_dotenv
from groq import AsyncGroq

from app.config import is_configured


class GroqWhisperTranscriber:
    """Transcribe one compressed WebM utterance, optionally with a language hint."""

    def __init__(
        self,
        *,
        language: str | None = None,
        model: str = "whisper-large-v3-turbo",
    ) -> None:
        load_dotenv()
        self._client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self._language = language
        self._model = model

    @staticmethod
    def is_configured() -> bool:
        """Report whether Groq speech-to-text can be used without exposing the key."""
        return is_configured("GROQ_API_KEY")

    async def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> str:
        """Return a trimmed multilingual transcript without forcing a language."""
        request = {
            "model": self._model,
            "file": ("utterance.webm", audio, mime_type),
            "response_format": "json",
            "temperature": 0.0,
        }
        if self._language is not None:
            request["language"] = self._language
        result = await self._client.audio.transcriptions.create(**request)
        return result.text.strip()

    async def close(self) -> None:
        """Release the HTTP client when a voice session ends."""
        await self._client.close()

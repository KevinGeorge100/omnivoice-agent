"""Cartesia streaming TTS bridge for OmniVoice."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from collections.abc import AsyncIterator, Iterable

import websockets

from app.config import is_configured


_DEFAULT_CARTESIA_URL = "wss://api.cartesia.ai/tts/websocket"


class CartesiaTTSStreamer:
    """Stream LLM text fragments to Cartesia and forward raw audio back to the client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        sample_rate: int = 16000,
        output_format: str = "pcm_s16le",
    ) -> None:
        self.api_key = (api_key or os.environ.get("CARTESIA_API_KEY") or "").strip()
        self.voice_id = (voice_id or os.environ.get("CARTESIA_VOICE_ID") or "").strip()
        self.model_id = (model_id or os.environ.get("CARTESIA_MODEL_ID") or "sonic-2").strip()
        self.sample_rate = int(sample_rate)
        self.output_format = output_format
        self._sequence = 0
        self._send_lock = asyncio.Lock()
        self._buffer: dict[str, str] = {}
        self._socket = None
        self._socket_lock = asyncio.Lock()

    @staticmethod
    def is_configured() -> bool:
        return is_configured("CARTESIA_API_KEY") and bool(os.environ.get("CARTESIA_VOICE_ID"))

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.voice_id)

    async def _get_connection(self):
        """Get or establish a long-lived persistent WebSocket connection to Cartesia."""
        if not self.configured:
            return None
        async with self._socket_lock:
            if self._socket is not None:
                is_open = False
                if hasattr(self._socket, "closed"):
                    is_open = not self._socket.closed
                elif hasattr(self._socket, "open"):
                    is_open = bool(self._socket.open)
                else:
                    is_open = True
                if is_open:
                    return self._socket

            headers = {"X-API-Key": self.api_key}
            try:
                self._socket = await websockets.connect(
                    _DEFAULT_CARTESIA_URL, additional_headers=headers
                )
                return self._socket
            except Exception:
                self._socket = None
                return None

    def _flush_trigger(self, text: str) -> bool:
        if len(text) >= 80 and any(token in text for token in (",", ";", ":")):
            return True
        if re.search(r"[.!?]$", text.strip()):
            return True
        if len(text) >= 140:
            return True
        return False

    def _trim_buffer(self, text: str) -> str:
        return " ".join(text.strip().split())

    async def process_token(self, token: str, websocket, *, turn_id: int | None = None) -> None:
        if not token:
            return
        key = str(turn_id) if turn_id is not None else "default"
        current = self._buffer.setdefault(key, "") + token
        self._buffer[key] = current
        clean = self._trim_buffer(current)
        if self._flush_trigger(clean):
            await self._synthesize_and_send(clean, websocket, turn_id=turn_id)
            self._buffer[key] = ""

    async def flush_remaining(self, websocket, *, turn_id: int | None = None) -> None:
        key = str(turn_id) if turn_id is not None else "default"
        remaining = self._trim_buffer(self._buffer.get(key, ""))
        if remaining:
            await self._synthesize_and_send(remaining, websocket, turn_id=turn_id)
            self._buffer[key] = ""

    async def send_audio_chunk(
        self,
        websocket,
        payload: bytes,
        *,
        turn_id: int | None = None,
        sequence_id: int | None = None,
    ) -> None:
        if not payload:
            return
        encoded = base64.b64encode(payload).decode("ascii")
        frame = {
            "type": "audio_chunk",
            "audio_b64": encoded,
            "sample_rate": self.sample_rate,
            "channels": 1,
            "format": self.output_format,
            "turn_id": turn_id,
            "sequence_id": self._sequence if sequence_id is None else sequence_id,
        }
        async with self._send_lock:
            await websocket.send_json(frame)
        if sequence_id is None:
            self._sequence += 1

    async def _stream_cartesia_audio(self, text: str) -> AsyncIterator[bytes]:
        if not text or not self.configured:
            return

        request = {
            "model_id": self.model_id,
            "voice": {"mode": "id", "id": self.voice_id},
            "text": text,
            "output_format": {
                "container": "raw",
                "encoding": self.output_format,
                "sample_rate": self.sample_rate,
            },
        }

        socket = await self._get_connection()
        if socket is None:
            return

        try:
            await socket.send(json.dumps(request))
            async for message in socket:
                if isinstance(message, bytes):
                    yield message
                    continue
                if not isinstance(message, str):
                    continue
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                packet = payload.get("audio") or payload.get("data")
                if packet:
                    try:
                        audio_bytes = base64.b64decode(packet)
                    except Exception:
                        audio_bytes = packet.encode("utf-8") if isinstance(packet, str) else b""
                    if audio_bytes:
                        yield audio_bytes
                if payload.get("type") in {"done", "final", "completed"}:
                    break
        except Exception:
            async with self._socket_lock:
                if self._socket is socket:
                    self._socket = None
            raise

    async def stream_tokens(
        self,
        token_iterable: Iterable[str] | AsyncIterator[str],
        websocket,
        *,
        turn_id: int | None = None,
    ) -> None:
        if not self.configured:
            return

        if hasattr(token_iterable, "__aiter__"):
            async for token in token_iterable:
                await self.process_token(token, websocket, turn_id=turn_id)
        else:
            for token in token_iterable:
                await self.process_token(token, websocket, turn_id=turn_id)

        await self.flush_remaining(websocket, turn_id=turn_id)

    async def _synthesize_and_send(self, text: str, websocket, *, turn_id: int | None = None) -> None:
        if not text or not self.configured:
            return
        try:
            async for audio in self._stream_cartesia_audio(text):
                if audio:
                    await self.send_audio_chunk(websocket, audio, turn_id=turn_id)
        except Exception:
            return

    async def close(self) -> None:
        """Close the persistent Cartesia WebSocket connection cleanly."""
        async with self._socket_lock:
            if self._socket is not None:
                try:
                    await self._socket.close()
                except Exception:
                    pass
                finally:
                    self._socket = None


cartesia_tts = CartesiaTTSStreamer()

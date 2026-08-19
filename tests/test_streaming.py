import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.main import stream_assistant_response


class RecordingWebSocket:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send_json(self, event: dict[str, object]) -> None:
        self.events.append(event)


class StreamingResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_hit_streams_answer_without_llm(self) -> None:
        websocket = RecordingWebSocket()
        with patch("app.main.semantic_cache.lookup", new=AsyncMock(return_value="Cached answer")):
            await stream_assistant_response(websocket, "business hours", asyncio.Lock(), turn_id=7)

        self.assertEqual(websocket.events[0]["type"], "assistant_token")
        self.assertEqual(websocket.events[0]["source"], "semantic_cache")
        self.assertEqual(websocket.events[0]["text"], "Cached answer")
        self.assertEqual(websocket.events[0]["turn_id"], 7)
        self.assertEqual(websocket.events[1]["type"], "assistant_response_end")
        self.assertEqual(websocket.events[1]["source"], "semantic_cache")
        self.assertEqual(websocket.events[1]["turn_id"], 7)

    async def test_missing_groq_key_returns_actionable_error(self) -> None:
        websocket = RecordingWebSocket()
        with (
            patch("app.main.semantic_cache.lookup", new=AsyncMock(return_value=None)),
            patch("app.main.ai_pipeline.is_configured", return_value=False),
        ):
            await stream_assistant_response(websocket, "uncached question", asyncio.Lock(), turn_id=3)

        self.assertEqual(websocket.events, [
            {
                "type": "assistant_error",
                "code": "groq_unconfigured",
                "message": "AI responses are unavailable. Configure GROQ_API_KEY and retry.",
                "turn_id": 3,
            }
        ])

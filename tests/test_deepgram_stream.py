import asyncio
import os
import unittest
from unittest.mock import patch

from app.deepgram_stream import DeepgramLiveTranscriber


class FakeConnection:
    def __init__(self) -> None:
        self._wait_forever = asyncio.Event()
        self.finalized = False
        self.closed = False

    async def recv(self) -> None:
        await self._wait_forever.wait()

    async def send_finalize(self) -> None:
        self.finalized = True

    async def send_close_stream(self) -> None:
        self.closed = True


class FakeConnectionContext:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.exited = False

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *_: object) -> None:
        self.exited = True


class FakeDeepgramClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.connection = FakeConnection()
        self.context = FakeConnectionContext(self.connection)
        self.listen = type(
            "Listen",
            (),
            {"v1": type("V1", (), {"connect": lambda _, **__: self.context})()},
        )()


class DeepgramStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_uses_configured_key_and_closes_cleanly(self) -> None:
        received: list[tuple[str, bool]] = []

        async def on_transcript(text: str, is_final: bool) -> None:
            received.append((text, is_final))

        with (
            patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test-deepgram-key"}),
            patch("app.deepgram_stream.AsyncDeepgramClient", FakeDeepgramClient),
        ):
            transcriber = DeepgramLiveTranscriber(on_transcript)
            await transcriber.start()
            connection = transcriber._connection
            self.assertIsNotNone(connection)
            await transcriber.close()

        self.assertEqual(received, [])
        self.assertTrue(connection.finalized)
        self.assertTrue(connection.closed)

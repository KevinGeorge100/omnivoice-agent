import asyncio
import unittest

from app.session import TranscriptDeduplicator, cancel_and_wait


class SessionGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_transcript_is_rejected_within_window(self) -> None:
        guard = TranscriptDeduplicator(window_seconds=10)

        self.assertTrue(guard.accept("Hello, OmniVoice"))
        self.assertFalse(guard.accept("  hello,   omnivoice  "))
        self.assertTrue(guard.accept("A different question"))

    async def test_cancel_and_wait_cleans_up_tasks(self) -> None:
        cancelled = asyncio.Event()

        async def wait_forever() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        tasks = {asyncio.create_task(wait_forever())}
        await asyncio.sleep(0)
        await cancel_and_wait(tasks)

        self.assertTrue(cancelled.is_set())
        self.assertFalse(tasks)

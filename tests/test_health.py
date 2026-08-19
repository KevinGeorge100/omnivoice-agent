import unittest

from httpx import ASGITransport, AsyncClient

from app.main import app


class HealthEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._previous_ready_state = app.state.semantic_cache_ready
        self._client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        app.state.semantic_cache_ready = self._previous_ready_state
        await self._client.aclose()

    async def test_health_never_exposes_secrets(self) -> None:
        response = await self._client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIsInstance(payload["active_connections"], int)
        self.assertEqual(set(payload["providers"]), {"groq", "deepgram", "sarvam"})
        self.assertTrue(all(isinstance(value, bool) for value in payload["providers"].values()))
        self.assertNotIn("API_KEY", response.text)

    async def test_readiness_reflects_semantic_cache_state(self) -> None:
        app.state.semantic_cache_ready = False
        warming_response = await self._client.get("/ready")
        self.assertEqual(warming_response.status_code, 503)
        self.assertEqual(warming_response.json()["status"], "warming")

        app.state.semantic_cache_ready = True
        ready_response = await self._client.get("/ready")
        self.assertEqual(ready_response.status_code, 200)
        self.assertEqual(ready_response.json()["status"], "ready")

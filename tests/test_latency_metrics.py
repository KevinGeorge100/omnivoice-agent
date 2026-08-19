import unittest

from app.latency_metrics import LatencyMetricsRegistry, TurnLatencyRecord


class LatencyMetricsTests(unittest.TestCase):
    def test_summary_exposes_recent_percentiles_without_transcripts(self) -> None:
        registry = LatencyMetricsRegistry(maximum_records=2)
        registry.record(
            TurnLatencyRecord(
                source="sarvam",
                cache_hit=False,
                stt_latency_ms=100,
                ttft_ms=200,
                turn_to_first_token_ms=300,
                response_latency_ms=400,
                full_turn_latency_ms=500,
            )
        )
        registry.record(
            TurnLatencyRecord(
                source="sarvam",
                cache_hit=False,
                stt_latency_ms=300,
                ttft_ms=400,
                turn_to_first_token_ms=700,
                response_latency_ms=800,
                full_turn_latency_ms=1_000,
            )
        )

        summary = registry.summary()

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["sources"], {"sarvam": 2})
        self.assertEqual(summary["metrics_ms"]["full_turn_latency"]["p95"], 1_000)

    def test_bounded_registry_discards_old_records(self) -> None:
        registry = LatencyMetricsRegistry(maximum_records=1)
        for source in ("groq_whisper", "sarvam"):
            registry.record(
                TurnLatencyRecord(source, False, None, None, None, None, None)
            )

        self.assertEqual(registry.summary()["sources"], {"sarvam": 1})

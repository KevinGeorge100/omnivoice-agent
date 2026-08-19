"""Bounded, in-memory latency observations for reproducible voice-turn analysis."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class TurnLatencyRecord:
    """One completed assistant turn; values are omitted when the stage was not observable."""

    source: str
    cache_hit: bool
    stt_latency_ms: float | None
    ttft_ms: float | None
    turn_to_first_token_ms: float | None
    response_latency_ms: float | None
    full_turn_latency_ms: float | None

    def payload(self) -> dict[str, object]:
        return asdict(self)


class LatencyMetricsRegistry:
    """Keep a fixed recent window and expose percentile summaries without caller content."""

    def __init__(self, maximum_records: int = 500) -> None:
        if maximum_records < 1:
            raise ValueError("maximum_records must be at least 1")
        self._records: deque[TurnLatencyRecord] = deque(maxlen=maximum_records)

    def record(self, record: TurnLatencyRecord) -> None:
        self._records.append(record)

    def summary(self) -> dict[str, object]:
        records = list(self._records)
        return {
            "sample_count": len(records),
            "metrics_ms": {
                "stt_latency": _distribution(record.stt_latency_ms for record in records),
                "time_to_first_token": _distribution(record.ttft_ms for record in records),
                "turn_to_first_token": _distribution(
                    record.turn_to_first_token_ms for record in records
                ),
                "response_latency": _distribution(record.response_latency_ms for record in records),
                "full_turn_latency": _distribution(
                    record.full_turn_latency_ms for record in records
                ),
            },
            "sources": _source_counts(records),
        }


def _distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    samples = sorted(value for value in values if value is not None)
    if not samples:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(samples),
        "p50": round(float(median(samples)), 2),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
    }


def _percentile(samples: list[float], quantile: float) -> float:
    index = max(0, math.ceil(len(samples) * quantile) - 1)
    return round(samples[index], 2)


def _source_counts(records: Iterable[TurnLatencyRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


latency_metrics = LatencyMetricsRegistry()

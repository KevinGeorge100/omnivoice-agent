"""Pure helpers for controlled speech-to-text benchmark scoring."""

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def normalize_transcript(text: str) -> str:
    """Normalize harmless formatting differences before accuracy scoring."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def edit_distance(expected: list[str], actual: list[str]) -> int:
    """Return a standard Levenshtein edit distance for token sequences."""
    previous = list(range(len(actual) + 1))
    for expected_index, expected_token in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_token in enumerate(actual, start=1):
            cost = 0 if expected_token == actual_token else 1
            current.append(
                min(
                    previous[actual_index] + 1,
                    current[actual_index - 1] + 1,
                    previous[actual_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def score_transcript(expected: str, actual: str) -> dict[str, float | int]:
    """Return character and word error rates for a reference/transcript pair."""
    normalized_expected = normalize_transcript(expected)
    normalized_actual = normalize_transcript(actual)
    expected_characters = list(normalized_expected.replace(" ", ""))
    actual_characters = list(normalized_actual.replace(" ", ""))
    expected_words = normalized_expected.split()
    actual_words = normalized_actual.split()

    character_distance = edit_distance(expected_characters, actual_characters)
    word_distance = edit_distance(expected_words, actual_words)
    return {
        "character_errors": character_distance,
        "character_error_rate": round(character_distance / max(len(expected_characters), 1), 4),
        "word_errors": word_distance,
        "word_error_rate": round(word_distance / max(len(expected_words), 1), 4),
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL benchmark manifest and reject incomplete case definitions."""
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on manifest line {line_number}") from error

        required = {"id", "audio_file", "expected_transcript", "language_code", "tags"}
        missing = sorted(required - case.keys())
        if missing:
            raise ValueError(f"Manifest case on line {line_number} is missing: {', '.join(missing)}")
        if not isinstance(case["tags"], list):
            raise ValueError(f"Manifest case on line {line_number} has non-list tags")
        cases.append(case)

    if not cases:
        raise ValueError("Benchmark manifest contains no cases")
    return cases


def select_cases(cases: Iterable[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Limit paid benchmark execution to a deliberate, small number of samples."""
    if maximum < 1:
        raise ValueError("max-cases must be at least 1")
    return list(cases)[:maximum]

from pathlib import Path
import unittest

from app.benchmark import load_manifest, normalize_transcript, score_transcript, select_cases


class BenchmarkTests(unittest.TestCase):
    def test_normalization_ignores_case_spacing_and_punctuation(self) -> None:
        self.assertEqual(normalize_transcript(" Hello,   OmniVoice! "), "hello omnivoice")

    def test_exact_transcript_has_zero_error_rates(self) -> None:
        score = score_transcript("നമസ്കാരം", "നമസ്കാരം")
        self.assertEqual(score["character_error_rate"], 0.0)
        self.assertEqual(score["word_error_rate"], 0.0)

    def test_manifest_is_credit_safe_and_case_capped(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = load_manifest(root / "benchmarks" / "malayalam_manifest.jsonl")

        self.assertEqual(len(cases), 8)
        self.assertEqual(len(select_cases(cases, 3)), 3)
        self.assertTrue(all(not case["expected_transcript"] for case in cases))

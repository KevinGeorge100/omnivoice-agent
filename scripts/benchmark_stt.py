"""Run a deliberate, credit-capped STT benchmark against local audio samples."""

import argparse
import asyncio
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq
from sarvamai import AsyncSarvamAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark import load_manifest, score_transcript, select_cases


DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "malayalam_manifest.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "benchmarks" / "results" / "stt_benchmark_results.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--providers", default="groq,sarvam", help="Comma-separated: groq,sarvam")
    parser.add_argument("--max-cases", type=int, default=3, help="Maximum samples to send per provider")
    parser.add_argument("--run", action="store_true", help="Allow provider API calls and credit usage")
    return parser.parse_args()


async def read_audio(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)


async def transcribe_groq(client: AsyncGroq, case: dict[str, Any], audio: bytes) -> str:
    mime_type = mimetypes.guess_type(case["audio_file"])[0] or "audio/wav"
    result = await client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=(Path(case["audio_file"]).name, audio, mime_type),
        language="ml",
        response_format="json",
        temperature=0.0,
    )
    return result.text.strip()


async def transcribe_sarvam(
    client: AsyncSarvamAI,
    case: dict[str, Any],
    audio: bytes,
) -> str:
    mime_type = mimetypes.guess_type(case["audio_file"])[0] or "audio/wav"
    result = await client.speech_to_text.transcribe(
        file=(Path(case["audio_file"]).name, audio, mime_type),
        model="saaras:v3",
        mode=case.get("sarvam_mode", "transcribe"),
        language_code=case["language_code"],
    )
    return result.transcript.strip()


async def evaluate_case(
    provider: str,
    case: dict[str, Any],
    groq_client: AsyncGroq | None,
    sarvam_client: AsyncSarvamAI | None,
) -> dict[str, Any]:
    audio_path = PROJECT_ROOT / "benchmarks" / case["audio_file"]
    if not case["expected_transcript"].strip():
        raise ValueError(f"Case {case['id']} has no expected_transcript")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file is missing for {case['id']}: {audio_path}")

    audio = await read_audio(audio_path)
    started_at = time.perf_counter()
    if provider == "groq":
        assert groq_client is not None
        transcript = await transcribe_groq(groq_client, case, audio)
    else:
        assert sarvam_client is not None
        transcript = await transcribe_sarvam(sarvam_client, case, audio)
    latency_ms = (time.perf_counter() - started_at) * 1000
    return {
        "case_id": case["id"],
        "provider": provider,
        "tags": case["tags"],
        "expected_transcript": case["expected_transcript"],
        "transcript": transcript,
        "latency_ms": round(latency_ms, 2),
        **score_transcript(case["expected_transcript"], transcript),
    }


async def main() -> None:
    arguments = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")
    cases = select_cases(load_manifest(arguments.manifest), arguments.max_cases)
    providers = [provider.strip().lower() for provider in arguments.providers.split(",") if provider.strip()]
    unknown = set(providers) - {"groq", "sarvam"}
    if unknown:
        raise ValueError(f"Unsupported providers: {', '.join(sorted(unknown))}")

    if not arguments.run:
        print(json.dumps({
            "mode": "dry-run",
            "providers": providers,
            "selected_cases": [case["id"] for case in cases],
            "message": "No provider APIs were called. Add consented audio and references, then rerun with --run.",
        }, ensure_ascii=False, indent=2))
        return

    required_keys = {
        "groq": "GROQ_API_KEY",
        "sarvam": "SARVAM_API_KEY",
    }
    missing_keys = [required_keys[provider] for provider in providers if not os.environ.get(required_keys[provider])]
    if missing_keys:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_keys)}")

    groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY")) if "groq" in providers else None
    sarvam_client = (
        AsyncSarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        if "sarvam" in providers
        else None
    )
    try:
        results = []
        for provider in providers:
            for case in cases:
                results.append(await evaluate_case(provider, case, groq_client, sarvam_client))
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            arguments.output.write_text,
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {len(results)} result(s) to {arguments.output}")
    finally:
        if groq_client is not None:
            await groq_client.close()


if __name__ == "__main__":
    asyncio.run(main())

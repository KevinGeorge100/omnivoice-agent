"""Create a small, transcript-backed Malayalam Common Voice benchmark subset."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mdc_benchmark import prepare_common_voice_benchmark


DOWNLOAD_DIRECTORY = PROJECT_ROOT / "benchmarks" / "downloads"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "benchmarks" / "generated" / "malayalam_mdc_test_manifest.jsonl"
DEFAULT_AUDIO_DIRECTORY = PROJECT_ROOT / "benchmarks" / "samples" / "mdc_common_voice_test"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Downloaded MDC .tar.gz archive; autodetected if omitted")
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--minimum-duration-ms", type=int, default=2_500)
    parser.add_argument("--maximum-duration-ms", type=int, default=15_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--audio-directory", type=Path, default=DEFAULT_AUDIO_DIRECTORY)
    return parser.parse_args()


def find_archive() -> Path:
    archives = sorted(DOWNLOAD_DIRECTORY.glob("*.tar.gz"))
    if len(archives) != 1:
        raise FileNotFoundError("Pass --archive explicitly; expected exactly one .tar.gz file in benchmarks/downloads/")
    return archives[0]


async def main() -> None:
    arguments = parse_arguments()
    archive = arguments.archive or find_archive()
    cases = await asyncio.to_thread(
        prepare_common_voice_benchmark,
        archive,
        arguments.output,
        arguments.audio_directory,
        split=arguments.split,
        count=arguments.sample_count,
        minimum_duration_ms=arguments.minimum_duration_ms,
        maximum_duration_ms=arguments.maximum_duration_ms,
    )
    print(
        json.dumps(
            {
                "prepared_cases": len(cases),
                "manifest": str(arguments.output),
                "audio_directory": str(arguments.audio_directory),
                "next_command": f"python scripts/benchmark_stt.py --manifest {arguments.output} --providers groq --run --max-cases 3",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

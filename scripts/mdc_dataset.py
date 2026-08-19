"""Inspect or explicitly download a Mozilla Data Collective dataset for benchmarking."""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mdc import MozillaDataCollectiveClient, download_dataset


MALAYALAM_COMMON_VOICE_DATASET_ID = "cmqiglff100innq07ytefmv64"
DEFAULT_DOWNLOAD_DIRECTORY = PROJECT_ROOT / "benchmarks" / "downloads"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=MALAYALAM_COMMON_VOICE_DATASET_ID)
    parser.add_argument("--download", action="store_true", help="Create a download session and fetch the archive")
    parser.add_argument(
        "--confirm-dataset-id",
        help="Required with --download; repeat the exact dataset ID to prevent accidental downloads",
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_DOWNLOAD_DIRECTORY)
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=256,
        help="Safety limit for a requested archive (default: 256 MB)",
    )
    return parser.parse_args()


async def main() -> None:
    arguments = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("MDC_API_KEY", "")
    if not api_key:
        raise RuntimeError("MDC_API_KEY is missing. Add it to .env; never commit the key.")

    async with MozillaDataCollectiveClient(api_key) as client:
        details = await client.get_dataset(arguments.dataset_id)
        print(json.dumps({"mode": "metadata", "dataset": asdict(details)}, ensure_ascii=False, indent=2))

        if not arguments.download:
            print("Metadata only: no download session was created and no archive was downloaded.")
            return
        if arguments.confirm_dataset_id != arguments.dataset_id:
            raise ValueError("--download requires --confirm-dataset-id with the exact selected dataset ID")

        maximum_bytes = arguments.max_size_mb * 1024 * 1024
        if details.size_bytes > maximum_bytes:
            raise ValueError(
                f"Dataset is {details.size_bytes:,} bytes; increase --max-size-mb deliberately to continue."
            )
        download = await client.create_download(arguments.dataset_id)
        destination = arguments.directory / download.filename
        downloaded_path = await download_dataset(download, destination, maximum_bytes=maximum_bytes)
        print(f"Downloaded and verified archive: {downloaded_path}")


if __name__ == "__main__":
    asyncio.run(main())

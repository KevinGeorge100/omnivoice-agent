"""Async client for safely retrieving Mozilla Data Collective benchmark data."""

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


MDC_API_BASE_URL = "https://mozilladatacollective.com/api"
DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class MozillaDataCollectiveError(RuntimeError):
    """Raised when Mozilla Data Collective rejects or cannot fulfill a request."""


def _as_positive_integer(value: object, field_name: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise MozillaDataCollectiveError(f"Invalid {field_name} returned by Mozilla Data Collective") from error
    if result < 0:
        raise MozillaDataCollectiveError(f"Invalid {field_name} returned by Mozilla Data Collective")
    return result


def _validate_dataset_id(dataset_id: str) -> str:
    if not DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("dataset_id must contain only letters, numbers, underscores, or hyphens")
    return dataset_id


@dataclass(frozen=True)
class DatasetDetails:
    """The benchmark-relevant subset of a Mozilla Data Collective dataset."""

    id: str
    name: str
    locale: str | None
    size_bytes: int
    license: str | None
    task: str | None
    file_format: str | None
    dataset_url: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DatasetDetails":
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            locale=payload.get("locale"),
            size_bytes=_as_positive_integer(payload.get("sizeBytes"), "sizeBytes"),
            license=payload.get("license"),
            task=payload.get("task"),
            file_format=payload.get("format"),
            dataset_url=payload.get("datasetUrl"),
        )


@dataclass(frozen=True)
class DatasetDownload:
    """A temporary, provider-issued download session."""

    url: str
    filename: str
    size_bytes: int
    checksum: str | None
    expires_at: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DatasetDownload":
        url = str(payload.get("downloadUrl", ""))
        if not url.startswith("https://"):
            raise MozillaDataCollectiveError("Mozilla Data Collective returned an invalid download URL")
        filename = Path(str(payload.get("filename", "dataset-download"))).name
        if filename in {"", "."}:
            raise MozillaDataCollectiveError("Mozilla Data Collective returned an invalid download filename")
        return cls(
            url=url,
            filename=filename,
            size_bytes=_as_positive_integer(payload.get("sizeBytes"), "sizeBytes"),
            checksum=payload.get("checksum"),
            expires_at=payload.get("expiresAt"),
        )


class MozillaDataCollectiveClient:
    """Small async REST client; it never creates a download session implicitly."""

    def __init__(self, api_key: str, *, timeout_seconds: float = 30.0) -> None:
        if not api_key.strip():
            raise ValueError("MDC_API_KEY is required")
        self._client = httpx.AsyncClient(
            base_url=MDC_API_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "MozillaDataCollectiveClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def get_dataset(self, dataset_id: str) -> DatasetDetails:
        """Fetch metadata only; this does not consume a dataset download allowance."""
        response = await self._client.get(f"/datasets/{_validate_dataset_id(dataset_id)}")
        payload = self._response_payload(response)
        return DatasetDetails.from_payload(payload)

    async def create_download(self, dataset_id: str) -> DatasetDownload:
        """Explicitly request a temporary direct-download URL after terms are accepted."""
        response = await self._client.post(f"/datasets/{_validate_dataset_id(dataset_id)}/download")
        payload = self._response_payload(response)
        return DatasetDownload.from_payload(payload)

    @staticmethod
    def _response_payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.is_error:
            detail = payload.get("error") if isinstance(payload, dict) else None
            suffix = f": {detail}" if detail else ""
            raise MozillaDataCollectiveError(f"MDC API request failed ({response.status_code}){suffix}")
        if not isinstance(payload, dict):
            raise MozillaDataCollectiveError("MDC API returned an unexpected response")
        return payload


async def download_dataset(
    download: DatasetDownload,
    destination: Path,
    *,
    maximum_bytes: int,
) -> Path:
    """Download a confirmed archive without blocking the event loop or overwriting files."""
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be at least 1")
    if download.size_bytes > maximum_bytes:
        raise MozillaDataCollectiveError(
            f"Dataset is {download.size_bytes:,} bytes, exceeding the {maximum_bytes:,}-byte safety limit"
        )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

    part_path = destination.with_name(f"{destination.name}.part")
    if part_path.exists():
        raise FileExistsError(f"Remove or rename the unfinished download first: {part_path}")

    await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
    downloaded_bytes = 0
    hasher = hashlib.sha256()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        ) as client:
            async with client.stream("GET", download.url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and _as_positive_integer(content_length, "content-length") > maximum_bytes:
                    raise MozillaDataCollectiveError("Download response exceeds the configured safety limit")
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > maximum_bytes:
                        raise MozillaDataCollectiveError("Download exceeded the configured safety limit")
                    await asyncio.to_thread(_append_bytes, part_path, chunk)
                    await asyncio.to_thread(hasher.update, chunk)

        if downloaded_bytes != download.size_bytes:
            raise MozillaDataCollectiveError(
                f"Downloaded {downloaded_bytes:,} bytes but MDC declared {download.size_bytes:,} bytes"
            )
        _verify_checksum(download.checksum, hasher.hexdigest())
        await asyncio.to_thread(os.replace, part_path, destination)
        return destination
    except BaseException:
        if part_path.exists():
            await asyncio.to_thread(part_path.unlink)
        raise


def _append_bytes(path: Path, chunk: bytes) -> None:
    with path.open("ab") as output_file:
        output_file.write(chunk)


def _verify_checksum(expected_checksum: str | None, actual_checksum: str) -> None:
    if not expected_checksum:
        return
    algorithm, separator, expected_digest = expected_checksum.partition(":")
    if algorithm.lower() == "sha256" and separator and expected_digest.lower() != actual_checksum.lower():
        raise MozillaDataCollectiveError("Downloaded archive failed its SHA-256 checksum verification")

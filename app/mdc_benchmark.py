"""Prepare a bounded, reproducible Common Voice evaluation subset from an MDC archive."""

import csv
import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import Iterable


@dataclass(frozen=True)
class CommonVoiceClip:
    """One transcript-backed audio clip eligible for an STT benchmark."""

    filename: str
    transcript: str
    speaker_id: str
    duration_ms: int


def select_common_voice_clips(
    rows: Iterable[dict[str, str]],
    durations: dict[str, int],
    *,
    count: int,
    minimum_duration_ms: int,
    maximum_duration_ms: int,
) -> list[CommonVoiceClip]:
    """Choose transcript-bearing clips deterministically while varying speakers first."""
    if count < 1:
        raise ValueError("sample count must be at least 1")
    if minimum_duration_ms < 0 or maximum_duration_ms < minimum_duration_ms:
        raise ValueError("invalid duration range")

    candidates: list[CommonVoiceClip] = []
    for row in rows:
        filename = Path(row.get("path", "")).name
        transcript = row.get("sentence", "").strip()
        duration_ms = durations.get(filename)
        if not filename or not transcript or duration_ms is None:
            continue
        if minimum_duration_ms <= duration_ms <= maximum_duration_ms:
            candidates.append(
                CommonVoiceClip(
                    filename=filename,
                    transcript=transcript,
                    speaker_id=row.get("client_id", "unknown"),
                    duration_ms=duration_ms,
                )
            )

    candidates.sort(key=lambda clip: hashlib.sha256(clip.filename.encode("utf-8")).hexdigest())
    selected: list[CommonVoiceClip] = []
    speaker_counts: Counter[str] = Counter()
    for maximum_per_speaker in (1, 2, 999_999):
        for clip in candidates:
            if clip in selected or speaker_counts[clip.speaker_id] >= maximum_per_speaker:
                continue
            selected.append(clip)
            speaker_counts[clip.speaker_id] += 1
            if len(selected) == count:
                return selected

    raise ValueError(
        f"Only {len(selected)} eligible clips found; adjust the duration range or lower --sample-count"
    )


def prepare_common_voice_benchmark(
    archive_path: Path,
    output_manifest: Path,
    audio_directory: Path,
    *,
    split: str,
    count: int,
    minimum_duration_ms: int,
    maximum_duration_ms: int,
) -> list[dict[str, object]]:
    """Extract a small known subset without ever using tarfile.extractall()."""
    if output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output_manifest}")
    if audio_directory.exists():
        raise FileExistsError(f"Refusing to overwrite audio directory: {audio_directory}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive is missing: {archive_path}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        root = _find_common_voice_root(archive, split)
        rows = _read_tsv(archive, f"{root}/{split}.tsv")
        durations = {
            row["clip"]: int(row["duration[ms]"])
            for row in _read_tsv(archive, f"{root}/clip_durations.tsv")
            if row.get("clip") and row.get("duration[ms]", "").isdigit()
        }
        selected = select_common_voice_clips(
            rows,
            durations,
            count=count,
            minimum_duration_ms=minimum_duration_ms,
            maximum_duration_ms=maximum_duration_ms,
        )

        staging_audio_directory = audio_directory.with_name(f"{audio_directory.name}.staging")
        staging_manifest = output_manifest.with_name(f"{output_manifest.name}.staging")
        if staging_audio_directory.exists() or staging_manifest.exists():
            raise FileExistsError("Remove the previous incomplete benchmark preparation before retrying")
        try:
            staging_audio_directory.mkdir(parents=True)
            manifest_cases = []
            for index, clip in enumerate(selected, start=1):
                _copy_archive_member(
                    archive,
                    f"{root}/clips/{clip.filename}",
                    staging_audio_directory / clip.filename,
                )
                manifest_cases.append(
                    {
                        "id": f"mdc_ml_{split}_{index:03d}",
                        "audio_file": str((audio_directory / clip.filename).relative_to(audio_directory.parents[1])).replace("\\", "/"),
                        "expected_transcript": clip.transcript,
                        "language_code": "ml-IN",
                        "sarvam_mode": "transcribe",
                        "tags": [
                            "malayalam",
                            "mozilla-data-collective",
                            "common-voice-scripted",
                            f"{split}-split",
                            f"duration-{clip.duration_ms}ms",
                        ],
                    }
                )
            output_manifest.parent.mkdir(parents=True, exist_ok=True)
            staging_manifest.write_text(
                "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in manifest_cases),
                encoding="utf-8",
            )
            os.replace(staging_audio_directory, audio_directory)
            os.replace(staging_manifest, output_manifest)
            return manifest_cases
        except BaseException:
            if staging_audio_directory.exists():
                shutil.rmtree(staging_audio_directory)
            if staging_manifest.exists():
                staging_manifest.unlink()
            raise


def _find_common_voice_root(archive: tarfile.TarFile, split: str) -> str:
    suffix = f"/ml/{split}.tsv"
    matches = [member.name[: -len(f"/{split}.tsv")] for member in archive.getmembers() if member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one Malayalam {split}.tsv manifest in the archive; found {len(matches)}")
    return matches[0]


def _read_tsv(archive: tarfile.TarFile, member_name: str) -> list[dict[str, str]]:
    member = archive.getmember(member_name)
    if not member.isfile():
        raise ValueError(f"Archive member is not a file: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Unable to read archive member: {member_name}")
    with source, TextIOWrapper(source, encoding="utf-8", newline="") as text_source:
        return list(csv.DictReader(text_source, delimiter="\t"))


def _copy_archive_member(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError("Unsafe archive member path")
    member = archive.getmember(member_name)
    if not member.isfile():
        raise ValueError(f"Expected an audio file in the archive: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Unable to read archive member: {member_name}")
    with source, destination.open("xb") as output_file:
        shutil.copyfileobj(source, output_file, length=1024 * 1024)

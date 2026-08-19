"""No-network tests for Mozilla Data Collective response and safety handling."""

import base64
import unittest

from app.mdc import (
    DatasetDetails,
    DatasetDownload,
    MozillaDataCollectiveError,
    _validate_dataset_id,
)
from app.mdc_benchmark import select_common_voice_clips
from app.sarvam_stream import SarvamLiveTranscriber


class MozillaDataCollectiveTests(unittest.TestCase):
    def test_dataset_details_parses_documented_payload(self) -> None:
        details = DatasetDetails.from_payload(
            {
                "id": "dataset-1",
                "name": "Common Voice",
                "locale": "ml",
                "sizeBytes": "1024",
                "license": "CC0-1.0",
                "task": "ASR",
                "format": "MP3",
                "datasetUrl": "https://example.test/datasets/dataset-1",
            }
        )

        self.assertEqual(details.size_bytes, 1024)
        self.assertEqual(details.locale, "ml")

    def test_download_rejects_non_https_url(self) -> None:
        with self.assertRaises(MozillaDataCollectiveError):
            DatasetDownload.from_payload(
                {"downloadUrl": "http://example.test/archive.tar.gz", "filename": "archive.tar.gz", "sizeBytes": "1"}
            )

    def test_dataset_id_cannot_contain_path_segments(self) -> None:
        with self.assertRaises(ValueError):
            _validate_dataset_id("../private")

    def test_clip_selection_is_bounded_and_prefers_speaker_variety(self) -> None:
        rows = [
            {"path": "one.mp3", "sentence": "ഒന്ന്", "client_id": "speaker-a"},
            {"path": "two.mp3", "sentence": "രണ്ട്", "client_id": "speaker-a"},
            {"path": "three.mp3", "sentence": "മൂന്ന്", "client_id": "speaker-b"},
        ]
        selected = select_common_voice_clips(
            rows,
            {"one.mp3": 3_000, "two.mp3": 4_000, "three.mp3": 5_000},
            count=2,
            minimum_duration_ms=2_500,
            maximum_duration_ms=15_000,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual({clip.speaker_id for clip in selected}, {"speaker-a", "speaker-b"})


class SarvamStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_pcm_frames_are_base64_encoded_for_the_provider_socket(self) -> None:
        captured: dict[str, object] = {}

        class FakeConnection:
            async def send_realtime_audio_input(self, message: object) -> None:
                captured["audio"] = getattr(message, "audio")

        async def noop_transcript(_: str, __: bool) -> None:
            return None

        async def noop_vad(_: str) -> None:
            return None

        async def noop_error(_: str) -> None:
            return None

        transcriber = SarvamLiveTranscriber(noop_transcript, noop_vad, noop_error)
        transcriber._connection = FakeConnection()
        await transcriber.send_pcm(b"\x00\x80\xff\x7f")

        self.assertEqual(base64.b64decode(str(captured["audio"])), b"\x00\x80\xff\x7f")

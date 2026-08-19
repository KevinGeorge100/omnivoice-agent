# OmniVoice Core

OmniVoice Core is a Python prototype for a real-time, full-duplex AI voice agent. It combines a FastAPI WebSocket transport, browser microphone capture, browser speech recognition and synthesis, a FAISS semantic cache, and streamed Groq responses.

## Project vision

OmniVoice is being developed as a **Real-Time AI Voice Agent Infrastructure**, not merely
as a speech-enabled chat application. Its governing goal is to implement the base paper's
WebSocket and connection-pooling architecture to achieve sub-500ms latency, introduce a
standardized conversational evaluation harness, and stream the open-source FD-Bench and
Fisher Corpus audio datasets through the architecture to create a mathematically provable
benchmark.

The full preserved vision, research scope, implementation path, and evidence requirements
are in [Project vision and research path](docs/project_vision.md). This document governs
future work: latency claims must be reproducible and dataset-driven.

## What it does

- Streams 100 ms binary WebM audio chunks over a WebSocket.
- Keeps the existing auto English/Malayalam Groq Whisper fallback for completed utterances.
- Transcribes English live server-side with Deepgram Nova-3 when selected.
- Streams Malayalam 16 kHz PCM frames to Sarvam Saaras v3 real-time STT when Malayalam is selected and Sarvam is configured.
- Uses browser speech recognition for the Malayalam-only fallback option.
- Streams LLM response tokens from Groq using `openai/gpt-oss-20b`.
- Checks a local semantic cache before calling the LLM.
- Speaks incoming responses sentence-by-sentence with browser speech synthesis.
- Supports barge-in: a new user utterance cancels the in-flight assistant response.
- Rejects immediate duplicate final transcripts and tags every assistant turn with a server-issued turn ID.
- Reconnects an interrupted browser session up to three times with exponential backoff.
- Displays connection state, STT latency, transport acknowledgements, cache hits, and time to first token (TTFT).

## Architecture

```text
Browser microphone / typed message
  -> WebSocket (/ws/audio/{client_id})
  -> Auto: browser VAD + Groq Whisper language detection
     OR English: Deepgram live transcription
     OR Malayalam: browser 16 kHz PCM + Sarvam streaming STT/VAD
  -> SemanticCacheRouter (FAISS + MiniLM)
  -> cached answer OR AsyncGroq stream
  -> WebSocket assistant tokens
  -> browser conversation log + speech synthesis
```

The semantic cache uses `sentence-transformers/all-MiniLM-L6-v2` with normalized embeddings and FAISS `IndexFlatIP`, which implements cosine-similarity lookup. Three business FAQs are seeded at startup: business hours, campus location, and contact information.

## Prerequisites

- Python 3.11 or later
- A Groq API key for answers that are not served by the semantic cache
- A Deepgram API key for server-side live transcription
- Chrome or Edge recommended for browser speech recognition

**Auto** mode sends a completed, voice-detected utterance to Whisper and attempts to identify English or Malayalam. Automatic identification can be unreliable for very short, noisy, or code-mixed speech. For Malayalam, choose **Malayalam — Sarvam live STT (recommended)**: it uses a persistent Saaras v3 real-time stream with 16 kHz PCM input, partial transcripts, and Sarvam server VAD. Groq Whisper remains a fallback if Sarvam cannot start. Select **English — Deepgram live STT** when English-only, low-latency interim captions are more important.

## Setup

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set your Groq key:

```env
GROQ_API_KEY=gsk_your_actual_key
DEEPGRAM_API_KEY=your_actual_deepgram_key
SARVAM_API_KEY=your_sarvam_key_for_future_malayalam_evaluation
MDC_API_KEY=your_mozilla_data_collective_key_for_optional_benchmark_data
```

Never commit `.env`. It is ignored by Git.

`SARVAM_API_KEY` is reserved for a later Malayalam streaming evaluation and is not used by the current runtime.

## Malayalam benchmark (credit-safe)

The repository includes an offline-first benchmark harness to compare Groq Whisper with Sarvam on the same consented Malayalam samples. It never calls either provider unless `--run` is explicitly added:

```powershell
python scripts/benchmark_stt.py
```

See [Malayalam benchmark guide](docs/malayalam_benchmark.md) before recording samples or spending Sarvam credits.

The optional Mozilla Data Collective integration is metadata-only by default and requires
an explicit, size-capped command before it downloads a benchmark archive. It is never
used by the live voice session. See the same guide for the safe download workflow.

After an approved dataset download, `scripts/prepare_mdc_benchmark.py` creates a small,
deterministic transcript-backed subset for a Groq-only first pass. Dataset archives,
extracted clips, and generated manifests remain outside Git.

This provider-selection benchmark is interim validation only. It does not replace the
project's required standardized conversational evaluation harness using FD-Bench and
Fisher Corpus; see [Project vision and research path](docs/project_vision.md).

## Run locally

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and choose **Start Voice Stream**. Grant microphone permission when prompted.

To make the app reachable from other devices on your local network, use `--host 0.0.0.0` and protect it appropriately before exposing it beyond a trusted network.

## Operational endpoints

- `GET /health` reports server liveness, active connection count, and whether each provider key is configured. It never returns secrets.
- `GET /ready` returns `200` after the semantic cache is warmed, or `503` while it is unavailable.
- `GET /metrics/latency` returns bounded, transcript-free p50/p95/p99 latency summaries for completed assistant turns. It currently measures STT latency when observable, time to first LLM token, and full-turn time; true Time-to-First-Audio will be added with provider-streamed TTS.

## Tests

Run the no-credit automated checks from the project root:

```powershell
python -m unittest discover -s tests -v
```

They cover provider-key validation, health/readiness responses, semantic-cache streaming, and the actionable missing-Groq error path. They do not call any external AI provider.

## WebSocket protocol

Connect to:

```text
ws://127.0.0.1:8000/ws/audio/{client_id}
```

### Client to server

| Type | Payload | Purpose |
| --- | --- | --- |
| Binary frame | WebM audio bytes | Sends a transport-baseline audio chunk. |
| Binary frame (`OVP1` prefix) | 16 kHz signed PCM bytes | Malayalam-only Sarvam streaming rail. The browser creates this in parallel; existing WebM transport remains unchanged. |
| `audio_utterance_start` | `{"type":"audio_utterance_start","mime_type":"audio/webm"}` followed by one binary frame | Sends a voice-detected complete utterance for auto-language transcription. |
| `user_utterance` | `{"type":"user_utterance","text":"..."}` | Sends a final transcript or typed message. |
| `barge_in` | `{"type":"barge_in"}` | Cancels active assistant response tasks. |

The WebSocket `language` query parameter is set by the UI. `auto` selects Groq Whisper language detection, `en-US` selects Deepgram live transcription, and `ml-IN` selects the Sarvam streaming Malayalam path. Groq Whisper is retained as the Malayalam fallback.

### Server to client

| Type | Key fields | Purpose |
| --- | --- | --- |
| Transport acknowledgement | `status`, `bytes_received`, `chunk_size`, `latency_ms` | Confirms received audio chunks. |
| `user_utterance_received` | `text`, `source`, `turn_id`, optional `stt_latency_ms` | Confirms receipt of a de-duplicated user utterance. |
| `stt_status` | `provider`, `ready` | Reports the selected transcription path. |
| `stt_transcript` | `text`, `is_final`, `source` | Sends interim or final transcript updates. |
| `stt_vad` | `source`, `signal` | Sarvam server VAD activity (`START_SPEECH` or `END_SPEECH`) for Malayalam streaming. |
| `assistant_token` | `text`, `source`, `turn_id`, optional `ttft_ms`, `turn_to_first_token_ms` | Streams assistant text. `source` is `semantic_cache` or `groq`. |
| `assistant_response_end` | `source`, `turn_id`, `total_response_ms`, optional `full_turn_latency_ms` | Marks completion of a response. |
| `assistant_interrupted` | `turn_id` | Confirms that barge-in cancelled the active response. |
| `assistant_error` | `message`, `code`, `turn_id` | Reports a generation error. |

Semantic-cache tokens also include `latency_tag: "<20ms"` for the UI.

## Project layout

```text
app/
  main.py              FastAPI app and full-duplex WebSocket endpoint
  ai_pipeline.py       AsyncGroq token streaming
  deepgram_stream.py   Async Deepgram Nova-3 streaming transcription
  groq_stt.py          Async Groq Whisper auto-language transcription/fallback
  sarvam_stream.py     Async Sarvam 16 kHz PCM Malayalam streaming bridge
  latency_metrics.py   Bounded percentile latency observations
  semantic_cache.py    MiniLM embeddings and FAISS cache router
  connection_pool.py   Active WebSocket connection tracking
static/
  index.html           Tailwind dashboard, audio capture, STT, and browser TTS
```

## Troubleshooting

- **The assistant returns an error for non-FAQ questions:** verify `GROQ_API_KEY`, then restart Uvicorn.
- **Auto mode does not transcribe:** grant microphone permission, speak after the one-second room-noise calibration, and pause briefly after each utterance. The browser sends the detected speech turn to Whisper after about 900 ms of silence.
- **Malayalam is inaccurate:** select **Malayalam — Sarvam live STT (recommended)** in a current Chrome or Edge browser. The server falls back to Groq Whisper only if Sarvam cannot start. Do not use Auto mode when Malayalam accuracy is required.
- **Malayalam streaming does not start:** verify `SARVAM_API_KEY`, then restart Uvicorn. This route requires browser AudioWorklet support; Chrome or Edge is recommended.
- **Deepgram cannot connect:** verify `DEEPGRAM_API_KEY`, then restart Uvicorn. The UI will use browser speech recognition as a fallback when supported.
- **Speech transcription does not start:** grant microphone permission. You can still use the typed message input.
- **No response after changing `.env`:** restart the server; environment variables are read at startup.
- **Model error from Groq:** update the model identifier in `app/ai_pipeline.py` to a model enabled for your Groq account.

## Security notes

- Keep API keys in `.env` or deployment-managed secrets only.
- Do not expose the local development server to the public internet without authentication, TLS, rate limits, and origin controls.
- Use separate API keys/projects for development, staging, and production.

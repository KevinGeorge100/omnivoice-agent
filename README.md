# OmniVoice Core

OmniVoice Core is a Python prototype for a real-time, full-duplex AI voice agent. It combines a FastAPI WebSocket transport, browser microphone capture, browser speech recognition and synthesis, a FAISS semantic cache, and streamed Groq responses.

## What it does

- Streams 100 ms binary WebM audio chunks over a WebSocket.
- Transcribes speech in supported browsers with the Web Speech API.
- Streams LLM response tokens from Groq using `llama-3.1-8b-instant`.
- Checks a local semantic cache before calling the LLM.
- Speaks incoming responses sentence-by-sentence with browser speech synthesis.
- Supports barge-in: a new user utterance cancels the in-flight assistant response.
- Displays connection state, transport acknowledgements, cache hits, and time to first token (TTFT).

## Architecture

```text
Browser microphone / typed message
  -> WebSocket (/ws/audio/{client_id})
  -> SemanticCacheRouter (FAISS + MiniLM)
  -> cached answer OR AsyncGroq stream
  -> WebSocket assistant tokens
  -> browser conversation log + speech synthesis
```

The semantic cache uses `sentence-transformers/all-MiniLM-L6-v2` with normalized embeddings and FAISS `IndexFlatIP`, which implements cosine-similarity lookup. Three business FAQs are seeded at startup: business hours, campus location, and contact information.

## Prerequisites

- Python 3.11 or later
- A Groq API key for answers that are not served by the semantic cache
- Chrome or Edge recommended for browser speech recognition

Deepgram is listed as an optional dependency for future server-side transcription. The current UI uses the browser Web Speech API, so a Deepgram key is not required to run the existing flow.

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
DEEPGRAM_API_KEY=your_deepgram_key_here
```

Never commit `.env`. It is ignored by Git.

## Run locally

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and choose **Start Voice Stream**. Grant microphone permission when prompted.

To make the app reachable from other devices on your local network, use `--host 0.0.0.0` and protect it appropriately before exposing it beyond a trusted network.

## WebSocket protocol

Connect to:

```text
ws://127.0.0.1:8000/ws/audio/{client_id}
```

### Client to server

| Type | Payload | Purpose |
| --- | --- | --- |
| Binary frame | Audio bytes | Sends an audio chunk. |
| `user_utterance` | `{"type":"user_utterance","text":"..."}` | Sends a final transcript or typed message. |
| `barge_in` | `{"type":"barge_in"}` | Cancels active assistant response tasks. |

### Server to client

| Type | Key fields | Purpose |
| --- | --- | --- |
| Transport acknowledgement | `status`, `bytes_received`, `chunk_size`, `latency_ms` | Confirms received audio chunks. |
| `user_utterance_received` | `text` | Confirms receipt of a user utterance. |
| `assistant_token` | `text`, `source`, optional `ttft_ms` | Streams assistant text. `source` is `semantic_cache` or `groq`. |
| `assistant_response_end` | `source` | Marks completion of a response. |
| `assistant_error` | `message` | Reports a generation error. |

Semantic-cache tokens also include `latency_tag: "<20ms"` for the UI.

## Project layout

```text
app/
  main.py              FastAPI app and full-duplex WebSocket endpoint
  ai_pipeline.py       AsyncGroq token streaming
  semantic_cache.py    MiniLM embeddings and FAISS cache router
  connection_pool.py   Active WebSocket connection tracking
static/
  index.html           Tailwind dashboard, audio capture, STT, and browser TTS
```

## Troubleshooting

- **The assistant returns an error for non-FAQ questions:** verify `GROQ_API_KEY`, then restart Uvicorn.
- **Speech transcription does not start:** grant microphone permission and use Chrome or Edge. You can still use the typed message input.
- **No response after changing `.env`:** restart the server; environment variables are read at startup.
- **Model error from Groq:** update the model identifier in `app/ai_pipeline.py` to a model enabled for your Groq account.

## Security notes

- Keep API keys in `.env` or deployment-managed secrets only.
- Do not expose the local development server to the public internet without authentication, TLS, rate limits, and origin controls.
- Use separate API keys/projects for development, staging, and production.

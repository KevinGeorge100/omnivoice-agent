import asyncio
import json
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from app.ai_pipeline import ai_pipeline, generate_llm_stream
from app.config import provider_status
from app.connection_pool import pool_manager
from app.deepgram_stream import DeepgramLiveTranscriber
from app.groq_stt import GroqWhisperTranscriber
from app.latency_metrics import TurnLatencyRecord, latency_metrics
from app.sarvam_stream import SarvamLiveTranscriber
from app.semantic_cache import semantic_cache
from app.session import TranscriptDeduplicator, cancel_and_wait

app = FastAPI(title="OmniVoice Core Transport Layer")
logger = logging.getLogger(__name__)
app.state.semantic_cache_ready = False
SARVAM_PCM_FRAME_PREFIX = b"OVP1"


@app.on_event("startup")
async def warm_semantic_cache() -> None:
    """Prepare seeded cache vectors before the first caller needs them."""
    try:
        await semantic_cache.warm()
        app.state.semantic_cache_ready = True
    except Exception:
        # The transport and cache-free LLM path can still run; readiness reports the issue.
        app.state.semantic_cache_ready = False
        logger.exception("Semantic cache warm-up failed")

    configured = provider_status()
    for provider_name, is_ready in configured.items():
        if not is_ready:
            logger.warning("%s API key is not configured", provider_name.capitalize())


@app.on_event("shutdown")
async def close_ai_clients() -> None:
    """Close application-level asynchronous clients cleanly."""
    await ai_pipeline.close()


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness and non-sensitive service configuration information."""
    return {
        "status": "ok",
        "active_connections": len(pool_manager.active_connections),
        "providers": provider_status(),
    }


@app.get("/ready")
async def readiness() -> JSONResponse:
    """Readiness probe: semantic cache must finish its startup warm-up."""
    is_ready = bool(app.state.semantic_cache_ready)
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "warming",
            "semantic_cache_ready": is_ready,
        },
    )


@app.get("/metrics/latency")
async def latency_summary() -> dict[str, object]:
    """Return recent non-sensitive latency distributions for the research harness."""
    return latency_metrics.summary()


async def stream_assistant_response(
    websocket: WebSocket,
    transcript: str,
    send_lock: asyncio.Lock,
    turn_id: int,
    *,
    turn_source: str = "browser",
    stt_latency_ms: float | None = None,
    turn_started_at: float | None = None,
) -> None:
    """Serve a semantic-cache hit or stream a fallback LLM response."""
    started_at = time.perf_counter()
    first_token_sent = False

    def turn_latency_ms() -> float | None:
        if turn_started_at is None:
            return None
        return round((time.perf_counter() - turn_started_at) * 1000, 2)

    def record_completion(
        *,
        cache_hit: bool,
        ttft_ms: float | None,
        response_latency_ms: float,
    ) -> None:
        latency_metrics.record(
            TurnLatencyRecord(
                source=turn_source,
                cache_hit=cache_hit,
                stt_latency_ms=stt_latency_ms,
                ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
                turn_to_first_token_ms=turn_latency_ms() if ttft_ms is not None else None,
                response_latency_ms=round(response_latency_ms, 2),
                full_turn_latency_ms=turn_latency_ms(),
            )
        )

    try:
        cached_response = await semantic_cache.lookup(transcript)
        if cached_response is not None:
            cache_latency_ms = (time.perf_counter() - started_at) * 1000
            logger.info("Semantic cache hit: %.2f ms", cache_latency_ms)
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "assistant_token",
                        "text": cached_response,
                        "source": "semantic_cache",
                        "latency_tag": "<20ms",
                        "ttft_ms": round(cache_latency_ms, 2),
                        "turn_id": turn_id,
                    }
                )
                await websocket.send_json(
                    {
                        "type": "assistant_response_end",
                        "source": "semantic_cache",
                        "turn_id": turn_id,
                        "total_response_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    }
                )
            record_completion(
                cache_hit=True,
                ttft_ms=cache_latency_ms,
                response_latency_ms=(time.perf_counter() - started_at) * 1000,
            )
            return

        if not ai_pipeline.is_configured():
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "assistant_error",
                        "code": "groq_unconfigured",
                        "message": "AI responses are unavailable. Configure GROQ_API_KEY and retry.",
                        "turn_id": turn_id,
                    }
                )
            return

        async for token in generate_llm_stream(transcript):
            is_first_token = not first_token_sent
            if not first_token_sent:
                ttft_ms = (time.perf_counter() - started_at) * 1000
                logger.info("LLM TTFT: %.2f ms", ttft_ms)
                first_token_sent = True

            async with send_lock:
                payload = {
                    "type": "assistant_token",
                    "text": token,
                    "source": "groq",
                    "turn_id": turn_id,
                }
                if is_first_token:
                    payload["ttft_ms"] = round(ttft_ms, 2)
                    if turn_started_at is not None:
                        payload["turn_to_first_token_ms"] = turn_latency_ms()
                await websocket.send_json(payload)

        response_latency_ms = (time.perf_counter() - started_at) * 1000
        async with send_lock:
            await websocket.send_json(
                {
                    "type": "assistant_response_end",
                    "source": "groq",
                    "turn_id": turn_id,
                    "total_response_ms": round(response_latency_ms, 2),
                    "full_turn_latency_ms": turn_latency_ms(),
                }
            )
        record_completion(
            cache_hit=False,
            ttft_ms=ttft_ms if first_token_sent else None,
            response_latency_ms=response_latency_ms,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Unable to stream assistant response")
        try:
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "assistant_error",
                        "code": "generation_failed",
                        "message": "Unable to generate a response. Please try again.",
                        "turn_id": turn_id,
                    }
                )
        except (RuntimeError, WebSocketDisconnect):
            # The client may have disconnected while the API call was in flight.
            pass

@app.websocket("/ws/audio/{client_id}")
async def audio_stream_endpoint(websocket: WebSocket, client_id: str):
    await pool_manager.connect(client_id, websocket)
    total_bytes_received = 0
    start_time = time.time()
    send_lock = asyncio.Lock()
    response_tasks: set[asyncio.Task[None]] = set()
    transcription_tasks: set[asyncio.Task[None]] = set()
    transcriber: DeepgramLiveTranscriber | None = None
    sarvam_transcriber: SarvamLiveTranscriber | None = None
    stt_language = websocket.query_params.get("language", "en-US")
    pending_utterance_mime: str | None = None
    pending_utterance_started_at: float | None = None
    whisper_transcriber: GroqWhisperTranscriber | None = None
    transcript_deduplicator = TranscriptDeduplicator()
    turn_id = 0
    sarvam_speech_started_at: float | None = None

    async def begin_assistant_response(
        transcript: str,
        source: str,
        stt_latency_ms: float | None = None,
        turn_started_at: float | None = None,
    ) -> None:
        """Cancel any active turn and start a response for one final transcript."""
        nonlocal turn_id
        if not transcript_deduplicator.accept(transcript):
            logger.info("Ignored duplicate transcript for client %s", client_id)
            return

        await cancel_and_wait(response_tasks)
        turn_id += 1

        async with send_lock:
            event = {
                "type": "user_utterance_received",
                "text": transcript,
                "source": source,
                "turn_id": turn_id,
            }
            if stt_latency_ms is not None:
                event["stt_latency_ms"] = round(stt_latency_ms, 2)
            await websocket.send_json(event)

        task = asyncio.create_task(
            stream_assistant_response(
                websocket,
                transcript,
                send_lock,
                turn_id,
                turn_source=source,
                stt_latency_ms=stt_latency_ms,
                turn_started_at=turn_started_at,
            ),
            name=f"llm-stream-{client_id}-turn-{turn_id}",
        )
        response_tasks.add(task)
        task.add_done_callback(response_tasks.discard)

    async def handle_deepgram_transcript(transcript: str, is_final: bool) -> None:
        """Relay interim text to the UI and route completed utterances to the agent."""
        async with send_lock:
            await websocket.send_json(
                {
                    "type": "stt_transcript",
                    "text": transcript,
                    "is_final": is_final,
                    "source": "deepgram",
                }
            )
        if is_final:
            await begin_assistant_response(transcript, source="deepgram")

    async def handle_sarvam_vad(signal_type: str) -> None:
        """Use Sarvam server VAD for a responsive Malayalam turn boundary and barge-in."""
        nonlocal sarvam_speech_started_at
        if signal_type == "START_SPEECH":
            sarvam_speech_started_at = time.perf_counter()
            await cancel_and_wait(response_tasks)
        async with send_lock:
            await websocket.send_json(
                {
                    "type": "stt_vad",
                    "source": "sarvam",
                    "signal": signal_type,
                }
            )

    async def handle_sarvam_transcript(
        transcript: str,
        is_final: bool,
    ) -> None:
        """Relay partial Saaras text and route completed turns into the cache/LLM pipeline."""
        if not is_final:
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_transcript",
                        "text": transcript,
                        "is_final": False,
                        "source": "sarvam",
                    }
                )
            return
        stt_latency_ms = (
            (time.perf_counter() - sarvam_speech_started_at) * 1000
            if sarvam_speech_started_at is not None
            else None
        )
        async with send_lock:
            await websocket.send_json(
                {
                    "type": "stt_transcript",
                    "text": transcript,
                    "is_final": True,
                    "source": "sarvam",
                }
            )
        await begin_assistant_response(
            transcript,
            source="sarvam",
            stt_latency_ms=stt_latency_ms,
            turn_started_at=sarvam_speech_started_at,
        )

    async def handle_sarvam_error(message: str) -> None:
        logger.warning("Sarvam streaming error for client %s: %s", client_id, message)
        try:
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_error",
                        "source": "sarvam",
                        "message": "Malayalam streaming transcription is unavailable. Try again or type your message.",
                    }
                )
        except (RuntimeError, WebSocketDisconnect):
            pass

    async def transcribe_auto_utterance(
        audio: bytes,
        mime_type: str,
        utterance_started_at: float,
    ) -> None:
        """Use Whisper to auto-detect and transcribe a completed speech turn."""
        try:
            if whisper_transcriber is None:
                raise RuntimeError("Auto speech transcriber is not initialized")
            transcript = await whisper_transcriber.transcribe(audio, mime_type)
            if not transcript:
                return
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_transcript",
                        "text": transcript,
                        "is_final": True,
                        "source": "groq_whisper",
                    }
                )
            stt_latency_ms = (time.perf_counter() - utterance_started_at) * 1000
            await begin_assistant_response(
                transcript,
                source="groq_whisper",
                stt_latency_ms=stt_latency_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Groq Whisper transcription failed for client %s", client_id)
            try:
                async with send_lock:
                    await websocket.send_json(
                        {
                            "type": "stt_error",
                            "message": "Unable to transcribe this utterance.",
                            "source": "groq_whisper",
                        }
                    )
            except (RuntimeError, WebSocketDisconnect):
                pass

    if stt_language.lower() == "auto":
        if GroqWhisperTranscriber.is_configured():
            whisper_transcriber = GroqWhisperTranscriber()
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_status",
                        "provider": "groq_whisper",
                        "ready": True,
                        "language": "auto",
                    }
                )
        else:
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_status",
                        "provider": "browser",
                        "ready": False,
                        "language": "en-US",
                        "message": "Groq speech-to-text is unavailable; using browser recognition.",
                    }
                )
    elif stt_language.lower() in {"ml", "ml-in"}:
        # Prefer the benchmark-selected Indic streaming provider. Groq remains a fallback.
        try:
            sarvam_transcriber = SarvamLiveTranscriber(
                handle_sarvam_transcript,
                handle_sarvam_vad,
                handle_sarvam_error,
            )
            await sarvam_transcriber.start()
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_status",
                        "provider": "sarvam",
                        "ready": True,
                        "language": "ml-IN",
                        "streaming": True,
                        "sample_rate": SarvamLiveTranscriber.sample_rate,
                    }
                )
        except Exception:
            logger.exception("Sarvam Malayalam streaming is unavailable for client %s", client_id)
            sarvam_transcriber = None
            if GroqWhisperTranscriber.is_configured():
                whisper_transcriber = GroqWhisperTranscriber(
                    language="ml",
                    model="whisper-large-v3",
                )
                async with send_lock:
                    await websocket.send_json(
                        {
                            "type": "stt_status",
                            "provider": "groq_whisper",
                            "ready": True,
                            "language": "ml",
                            "message": "Sarvam streaming is unavailable; using Groq Whisper fallback.",
                        }
                    )
            else:
                async with send_lock:
                    await websocket.send_json(
                        {
                            "type": "stt_status",
                            "provider": "browser",
                            "ready": False,
                            "language": "ml-IN",
                            "message": "Malayalam STT is unavailable; type a message or configure Sarvam.",
                        }
                    )
    else:
        try:
            transcriber = DeepgramLiveTranscriber(handle_deepgram_transcript)
            await transcriber.start()
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_status",
                        "provider": "deepgram",
                        "ready": True,
                        "language": stt_language,
                    }
                )
        except Exception:
            logger.exception("Deepgram streaming is unavailable for client %s", client_id)
            transcriber = None
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "stt_status",
                        "provider": "browser",
                        "ready": False,
                        "language": stt_language,
                    }
                )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    event = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Ignoring invalid JSON event from client %s", client_id)
                    async with send_lock:
                        await websocket.send_json(
                            {
                                "type": "client_error",
                                "code": "invalid_json",
                                "message": "The server received an invalid control event.",
                            }
                        )
                    continue

                event_type = event.get("type")

                if event_type == "barge_in":
                    logger.info("Barge-in received from client %s. Cancelling response tasks.", client_id)
                    await cancel_and_wait(response_tasks)
                    await cancel_and_wait(transcription_tasks)
                    async with send_lock:
                        await websocket.send_json(
                            {"type": "assistant_interrupted", "turn_id": turn_id}
                        )
                    continue

                if event_type == "audio_utterance_start":
                    mime_type = event.get("mime_type", "audio/webm")
                    if mime_type == "audio/webm":
                        pending_utterance_mime = mime_type
                        pending_utterance_started_at = time.perf_counter()
                    continue

                if event_type == "user_utterance" and isinstance(event.get("text"), str):
                    transcript = event["text"].strip()
                    if transcript:
                        await begin_assistant_response(transcript, source="browser")
                continue

            audio_chunk = message.get("bytes")
            if audio_chunk is None:
                continue
            if audio_chunk.startswith(SARVAM_PCM_FRAME_PREFIX):
                if sarvam_transcriber is not None:
                    await sarvam_transcriber.send_pcm(audio_chunk[len(SARVAM_PCM_FRAME_PREFIX) :])
                continue
            if pending_utterance_mime is not None:
                mime_type = pending_utterance_mime
                pending_utterance_mime = None
                utterance_started_at = pending_utterance_started_at or time.perf_counter()
                pending_utterance_started_at = None
                await cancel_and_wait(transcription_tasks)
                task = asyncio.create_task(
                    transcribe_auto_utterance(audio_chunk, mime_type, utterance_started_at),
                    name=f"whisper-stt-{client_id}",
                )
                transcription_tasks.add(task)
                task.add_done_callback(transcription_tasks.discard)
                continue
            if transcriber is not None:
                await transcriber.send_audio(audio_chunk)
            chunk_size = len(audio_chunk)
            total_bytes_received += chunk_size

            elapsed_ms = (time.time() - start_time) * 1000

            async with send_lock:
                await websocket.send_text(
                    f'{{"status": "streaming", "bytes_received": {total_bytes_received}, "chunk_size": {chunk_size}, "latency_ms": {round(elapsed_ms, 2)}}}'
                )
            start_time = time.time()

    except WebSocketDisconnect:
        pass
    finally:
        if transcriber is not None:
            await transcriber.close()
        if sarvam_transcriber is not None:
            await sarvam_transcriber.close()
        await cancel_and_wait(transcription_tasks)
        if whisper_transcriber is not None:
            await whisper_transcriber.close()
        await cancel_and_wait(response_tasks)
        pool_manager.disconnect(client_id)

@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

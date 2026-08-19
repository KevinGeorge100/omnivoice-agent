import asyncio
import json
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from app.ai_pipeline import generate_llm_stream
from app.connection_pool import pool_manager
from app.semantic_cache import semantic_cache

app = FastAPI(title="OmniVoice Core Transport Layer")
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def warm_semantic_cache() -> None:
    """Prepare seeded cache vectors before the first caller needs them."""
    await semantic_cache.warm()


async def stream_assistant_response(
    websocket: WebSocket,
    transcript: str,
    send_lock: asyncio.Lock,
) -> None:
    """Serve a semantic-cache hit or stream a fallback LLM response."""
    started_at = time.perf_counter()
    first_token_sent = False

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
                    }
                )
                await websocket.send_json(
                    {"type": "assistant_response_end", "source": "semantic_cache"}
                )
            return

        async for token in generate_llm_stream(transcript):
            is_first_token = not first_token_sent
            if not first_token_sent:
                ttft_ms = (time.perf_counter() - started_at) * 1000
                logger.info("LLM TTFT: %.2f ms", ttft_ms)
                first_token_sent = True

            async with send_lock:
                payload = {"type": "assistant_token", "text": token, "source": "groq"}
                if is_first_token:
                    payload["ttft_ms"] = round(ttft_ms, 2)
                await websocket.send_json(payload)

        async with send_lock:
            await websocket.send_json({"type": "assistant_response_end", "source": "groq"})
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Unable to stream assistant response")
        try:
            async with send_lock:
                await websocket.send_json(
                    {"type": "assistant_error", "message": "Unable to generate a response."}
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
                    continue

                event_type = event.get("type")

                if event_type == "barge_in":
                    logger.info("Barge-in received from client %s. Cancelling response tasks.", client_id)
                    for running_task in list(response_tasks):
                        running_task.cancel()
                    response_tasks.clear()
                    continue

                if event_type == "user_utterance" and isinstance(event.get("text"), str):
                    transcript = event["text"].strip()
                    if transcript:
                        # Cancel active response tasks for barge-in / full-duplex interruption
                        for running_task in list(response_tasks):
                            running_task.cancel()
                        response_tasks.clear()

                        async with send_lock:
                            await websocket.send_json(
                                {"type": "user_utterance_received", "text": transcript}
                            )

                        task = asyncio.create_task(
                            stream_assistant_response(websocket, transcript, send_lock),
                            name=f"llm-stream-{client_id}",
                        )
                        response_tasks.add(task)
                        task.add_done_callback(response_tasks.discard)
                continue

            audio_chunk = message.get("bytes")
            if audio_chunk is None:
                continue
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
        for task in response_tasks:
            task.cancel()
        if response_tasks:
            await asyncio.gather(*response_tasks, return_exceptions=True)
        pool_manager.disconnect(client_id)

@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

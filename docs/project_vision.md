# OmniVoice project vision and research path

This document preserves the project direction stated in Kevin George V's seminar
presentation. It is the source of truth for the work that follows; implementation
details may evolve, but the vision and contribution below must not be reworded or
narrowed.

## Vision and contribution — preserved verbatim

> **OmniVoice: A Real-Time AI Voice Agent Infrastructure**
>
> **Extending the Architecture:** I will implement the base paper's exact WebSocket and
> connection-pooling architecture to achieve sub-500ms latency.
>
> **Addressing the Limitation:** I will introduce a standardized conversational evaluation
> harness.
>
> **The Contribution:** Streaming the open-source FD-Bench and Fisher Corpus audio datasets
> through this architecture to create a mathematically provable benchmark.

## What this means in practice

OmniVoice is not only a chat interface with speech recognition. It is an experimental
real-time voice-agent infrastructure whose claims must be measured under repeatable
streaming conditions.

The intended system path is:

1. **Full-duplex transport foundation** — maintain persistent WebSocket sessions, stream
   audio in small packets, support interruption, and keep the async event loop free of
   blocking provider or file operations.
2. **Latency-first orchestration** — measure end-to-end Time-to-First-Audio (TTFA),
   turn-completion time, jitter, packet/connection behaviour, and percentile latency;
   do not rely on a single average or UI-only timing.
3. **Real conversational pipeline** — assemble asynchronous STT, LLM, and streaming TTS
   stages with context management, semantic caching, turn-taking, barge-in, and
   provider-specific fallbacks.
4. **Standardized conversational evaluation harness** — replay permitted FD-Bench and
   Fisher Corpus conversation audio through the same streaming transport and emit a
   reproducible benchmark report. Dataset licenses, access terms, and privacy restrictions
   must be observed; dataset media must not be committed or re-hosted.
5. **Evidence-based optimisation** — compare a sequential/reference path with the
   connection-pooled asynchronous path, state the exact configuration, and report the
   measurable effect on TTFA and complete-turn latency.

## Current implementation position

The project has established its initial WebSocket transport, async response streaming,
semantic cache, interruption handling, reconnect behaviour, provider health visibility,
and bounded p50/p95/p99 latency observation. Malayalam mode now has an opt-in Sarvam
16 kHz PCM streaming rail with server VAD while retaining the WebM transport baseline.
The Malayalam Common Voice/Mozilla Data Collective benchmark is an **interim STT-provider
validation**: it helps choose a transcription provider for Malayalam and does not replace
the required FD-Bench and Fisher conversational benchmark.

The next major implementation milestone is therefore to introduce provider-streamed TTS
and true Time-to-First-Audio measurement, then begin the dataset-adapter/replay design for
the standardized conversational evaluation harness. Live Malayalam routing remains a
supporting capability, not a substitute for the research contribution.

## Definition of a credible result

The project can claim a latency improvement only when a versioned benchmark run records:

- corpus and permitted split, scenario, audio packet size, and replay method;
- STT, LLM, and TTS providers/models plus region/configuration;
- connection-pooling and cache state;
- sample count and p50/p95/p99 TTFA and full-turn latency;
- failures, reconnects, interruptions, and exclusion criteria;
- the sequential/reference comparison under equivalent conditions.

This keeps the final result mathematically auditable rather than relying on a demo alone.

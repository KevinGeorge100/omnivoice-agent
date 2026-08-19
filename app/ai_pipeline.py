"""Async streaming LLM support for OmniVoice."""

from collections.abc import AsyncIterator
import os

from dotenv import load_dotenv
from groq import AsyncGroq

from app.config import is_configured


SYSTEM_PROMPT = (
    "You are OmniVoice, a helpful business voice assistant. "
    "Give concise, natural, action-oriented answers. "
    "Reply in the same language as the user whenever possible."
)


class StreamingAIPipeline:
    """Streams concise assistant responses from Groq without blocking the event loop."""

    def __init__(self) -> None:
        load_dotenv()
        self._client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self._model = "openai/gpt-oss-20b"

    @staticmethod
    def is_configured() -> bool:
        """Report whether LLM streaming can be used without revealing the API key."""
        return is_configured("GROQ_API_KEY")

    async def generate_llm_stream(self, transcript: str) -> AsyncIterator[str]:
        """Yield non-empty response tokens as they arrive from the LLM."""
        stream = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def close(self) -> None:
        """Release the async HTTP client when the application shuts down."""
        await self._client.close()


ai_pipeline = StreamingAIPipeline()


async def generate_llm_stream(transcript: str) -> AsyncIterator[str]:
    """Stream tokens through the shared OmniVoice LLM pipeline."""
    async for token in ai_pipeline.generate_llm_stream(transcript):
        yield token

"""FAISS-backed semantic response cache for common OmniVoice questions."""

import asyncio
from collections.abc import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCacheRouter:
    """Route semantically similar queries to cached answers using cosine similarity."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self.similarity_threshold = similarity_threshold
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._responses: list[str] = []
        self._initialize_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self._seed_entries = (
            ("What are your business hours?", "Our business hours are Monday through Friday, 9 AM to 5 PM."),
            ("Where is your campus located?", "Our campus is located at 100 Innovation Drive, downtown."),
            ("How can I contact your team?", "You can contact our team at support@omnivoice.example or (555) 010-2000."),
        )

    async def _initialize(self) -> None:
        """Load the embedding model and add the FAQ seed entries once."""
        if self._model is not None:
            return

        async with self._initialize_lock:
            if self._model is not None:
                return

            model = await asyncio.to_thread(SentenceTransformer, self.MODEL_NAME)
            seed_queries = [query for query, _ in self._seed_entries]
            vectors = await self._encode_with_model(model, seed_queries)

            self._model = model
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
            self._responses.extend(response for _, response in self._seed_entries)

    @staticmethod
    async def _encode_with_model(
        model: SentenceTransformer, queries: Sequence[str],
    ) -> np.ndarray:
        """Run synchronous embedding inference outside the AsyncIO event loop."""
        vectors = await asyncio.to_thread(
            model.encode,
            list(queries),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    async def _encode(self, queries: Sequence[str]) -> np.ndarray:
        await self._initialize()
        assert self._model is not None
        return await self._encode_with_model(self._model, queries)

    async def warm(self) -> None:
        """Preload the model and the seeded FAQ vectors during application startup."""
        await self._initialize()

    async def add_to_cache(self, query: str, response: str) -> None:
        """Add a query and its response to the semantic cache."""
        vector = await self._encode([query])
        async with self._index_lock:
            assert self._index is not None
            self._index.add(vector)
            self._responses.append(response)

    async def lookup(self, query: str) -> str | None:
        """Return the nearest cached response when it meets the similarity threshold."""
        vector = await self._encode([query])
        async with self._index_lock:
            assert self._index is not None
            scores, indexes = self._index.search(vector, 1)
            similarity = float(scores[0][0])
            response_index = int(indexes[0][0])

            if similarity < self.similarity_threshold or response_index < 0:
                return None
            return self._responses[response_index]


semantic_cache = SemanticCacheRouter()

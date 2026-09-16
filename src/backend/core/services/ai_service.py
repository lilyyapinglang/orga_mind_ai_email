"""Albert API client for AI features and grounded reply generation."""

import logging
from dataclasses import dataclass

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from openai import OpenAI

from core.ai.utils import is_ai_enabled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagChunk:
    """A source excerpt returned by Albert, with its audit metadata."""

    content: str
    chunk_id: int
    document_id: int
    collection_id: int
    search_score: float
    rerank_score: float | None = None

    def as_metadata(self) -> dict:
        """Return serializable source provenance for the generated draft."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "collection_id": self.collection_id,
            "search_score": self.search_score,
            "rerank_score": self.rerank_score,
            "excerpt": self.content,
        }


class AIService:
    """Client for Albert's OpenAI-compatible chat and RAG endpoints."""

    def __init__(self):
        if not is_ai_enabled():
            raise ImproperlyConfigured("AI configuration not set")
        self.base_url = settings.AI_BASE_URL.rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.AI_API_KEY}"}
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=settings.AI_API_KEY,
            timeout=60,
            max_retries=1,
        )

    def call_ai_api(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call Albert chat completions and return its non-empty text content."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=settings.AI_MODEL,
            messages=messages,
            stream=False,
            n=1,
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("AI response does not contain an answer")
        return response.choices[0].message.content

    @staticmethod
    def _collection_ids() -> list[int]:
        """Return configured public and private collections without mutation."""
        raw_ids = list(settings.AI_COLLECTION_IDS)
        if settings.AI_PRIVATE_COLLECTION_ID:
            raw_ids.append(settings.AI_PRIVATE_COLLECTION_ID)
        try:
            collection_ids = [int(value) for value in raw_ids]
        except (TypeError, ValueError) as exc:
            raise ImproperlyConfigured(
                "AI_COLLECTION_IDS and AI_PRIVATE_COLLECTION_ID must be integers"
            ) from exc
        return list(dict.fromkeys(collection_ids))

    @classmethod
    def _collection_ids_for_query(cls, query: str) -> list[int]:
        """Narrow retrieval when an administrator configured a topic route.

        Routes are deliberately declarative rather than guessed by an LLM. Each
        entry is ``{"keywords": [...], "collection_ids": [...]}``.  The
        private collection remains available for local procedures in every
        route; without a match the configured full allowlist is used.
        """
        all_ids = cls._collection_ids()
        query = query.casefold()
        routes = settings.AI_RAG_COLLECTION_ROUTES
        if not isinstance(routes, list):
            raise ImproperlyConfigured("AI_RAG_COLLECTION_ROUTES must be a JSON list")
        for route in routes:
            if not isinstance(route, dict):
                continue
            keywords = route.get("keywords", [])
            route_ids = route.get("collection_ids", [])
            if not isinstance(keywords, list) or not isinstance(route_ids, list):
                continue
            if any(isinstance(keyword, str) and keyword.casefold() in query for keyword in keywords):
                try:
                    selected = [int(value) for value in route_ids]
                except (TypeError, ValueError) as exc:
                    raise ImproperlyConfigured(
                        "AI_RAG_COLLECTION_ROUTES collection_ids must be integers"
                    ) from exc
                if settings.AI_PRIVATE_COLLECTION_ID:
                    selected.append(int(settings.AI_PRIVATE_COLLECTION_ID))
                return list(dict.fromkeys(selected))
        return all_ids

    def search_chunks(self, query: str) -> list[RagChunk]:
        """Retrieve a broad candidate set, scoped to known collections."""
        query = (query or "").strip()
        if not query:
            return []
        collection_ids = self._collection_ids_for_query(query)
        if not collection_ids:
            raise ImproperlyConfigured("At least one Albert RAG collection is required")
        response = requests.post(
            f"{self.base_url}/search",
            headers=self.headers,
            json={
                "query": query[: settings.AI_QUERY_MAX_CHARS],
                "collection_ids": collection_ids,
                "method": settings.AI_SEARCH_METHOD,
                "limit": settings.AI_RAG_CANDIDATE_LIMIT,
            },
            timeout=60,
        )
        response.raise_for_status()
        chunks = []
        for result in response.json().get("data", []):
            chunk = result.get("chunk", {}) if isinstance(result, dict) else {}
            try:
                content = chunk["content"].strip()
                if content:
                    chunks.append(
                        RagChunk(
                            content=content,
                            chunk_id=int(chunk["id"]),
                            document_id=int(chunk["document_id"]),
                            collection_id=int(chunk["collection_id"]),
                            search_score=float(result["score"]),
                        )
                    )
            except (KeyError, TypeError, ValueError, AttributeError):
                logger.warning("Ignoring malformed Albert search result")
        return chunks

    def rerank_chunks(self, query: str, chunks: list[RagChunk]) -> list[RagChunk]:
        """Keep only strongly relevant excerpts from a broad search result."""
        if not chunks:
            return []
        response = requests.post(
            f"{self.base_url}/rerank",
            headers=self.headers,
            json={
                "model": settings.AI_RAG_RERANK_MODEL,
                "query": query,
                "documents": [chunk.content for chunk in chunks],
                "top_n": settings.AI_RAG_CONTEXT_LIMIT,
            },
            timeout=60,
        )
        response.raise_for_status()
        ranked = []
        for result in response.json().get("results", []):
            try:
                index = int(result["index"])
                relevance = float(result["relevance_score"])
                if relevance < settings.AI_RAG_MIN_RELEVANCE:
                    continue
                chunk = chunks[index]
                ranked.append(
                    RagChunk(
                        content=chunk.content,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        collection_id=chunk.collection_id,
                        search_score=chunk.search_score,
                        rerank_score=relevance,
                    )
                )
            except (KeyError, TypeError, ValueError, IndexError):
                logger.warning("Ignoring malformed Albert rerank result")
        return ranked

    def retrieve_grounded_chunks(self, query: str) -> list[RagChunk]:
        """Search broadly then rerank, returning only usable evidence."""
        return self.rerank_chunks(query, self.search_chunks(query))

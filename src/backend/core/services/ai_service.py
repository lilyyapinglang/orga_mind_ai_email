"""Service for AI-powered features using OpenAI-compatible API."""
import mimetypes
import logging
import requests
import json
import os
import ast

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from openai import OpenAI

from core.ai.utils import is_ai_enabled

logger = logging.getLogger(__name__)

# Extensions et types MIME acceptés par POST /v1/documents
ALLOWED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 Mo

class AIService:
    """Service class for AI-related operations."""

    def __init__(self):
        """Ensure that the AI configuration is set properly."""
        self.headers = {}
        if not is_ai_enabled():
            raise ImproperlyConfigured("AI configuration not set")
        self.client = OpenAI(
            base_url=settings.AI_BASE_URL,
            api_key=settings.AI_API_KEY,
            timeout=60,
            max_retries=1,
        )
        logger.info(f"settings data: {settings}")
        self.__set_headers()


    def __set_headers(self) -> dict:
        """Build API auth headers from settings/environment.

        Raises ImproperlyConfigured so the caller can fall back gracefully.
        """
        api_key = settings.AI_API_KEY
        if not api_key:
            raise ImproperlyConfigured(
                "AI_API_KEY is not configured (settings.AI_API_KEY or env var)."
            )
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def __check_response(self, response: requests.Response) -> None:
        """Raise with the API's error detail included (a bare 422 hides the cause)."""
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Le corps de la réponse 422 indique le champ invalide, ex. :
            # {"detail":[{"type":"string_too_short","loc":["body","query"], ...}]}
            logger.error(
                "Albert API error %s on %s: %s",
                response.status_code,
                response.url,
                response.text[:1000],
            )
            raise requests.HTTPError(
                f"Albert API {response.status_code} on {response.url}: {response.text[:500]}"
            ) from exc

    @staticmethod
    def _collection_ids(collection_ids: list | None = None) -> list:
        """Return the collection ids to search, without mutating settings.

        Ids are passed through with their original type (int or str): Albert
        rejects the wrong type, and env parsing may produce either.

        ``settings.AI_COLLECTION_IDS`` is a shared list: appending the private
        collection to it directly made it grow on every single request.
        """
        if collection_ids is not None:
            return [cid for cid in collection_ids if cid]
        ids = list(settings.AI_COLLECTION_IDS or [])
        if settings.AI_PRIVATE_COLLECTION_ID:
            ids.append(settings.AI_PRIVATE_COLLECTION_ID)
        return [cid for cid in ids if cid]

    @staticmethod
    def _normalize_search_result(result: dict) -> dict | None:
        """Flatten one Albert search hit into a chunk dict carrying provenance.

        Keeping the collection/document/chunk ids (and not only the text) is
        what makes a generated draft auditable afterwards.
        """
        chunk = result.get("chunk") or {}
        content = (chunk.get("content") or "").strip()
        if not content:
            return None
        metadata = chunk.get("metadata") or {}
        return {
            "content": content,
            "chunk_id": chunk.get("id"),
            "document_id": chunk.get("document_id") or metadata.get("document_id"),
            "collection_id": chunk.get("collection_id")
            or metadata.get("collection_id"),
            "document_name": metadata.get("document_name") or metadata.get("name"),
            "search_score": result.get("score"),
            "search_method": result.get("method"),
            "rerank_score": None,
        }

    def search_chunks(
        self,
        question: str,
        limit: int | None = None,
        collection_ids: list | None = None,
    ) -> list[dict]:
        """Search relevant chunks in the Albert API vector store.

        Returns a list of normalized chunk dicts (text + provenance + score),
        ordered as returned by Albert.

        The query must be a non-empty, non-whitespace string, otherwise the API
        answers 422 Unprocessable Entity.
        """
        question = (question or "").strip()
        if not question:
            # Ne jamais appeler l'API avec une requête vide : c'est un 422 garanti.
            return []

        payload = {
            "query": question[: settings.AI_QUERY_MAX_CHARS],
            "method": settings.AI_SEARCH_METHOD or "semantic",
            "limit": limit or settings.AI_SEARCH_LIMIT,
        }
        ids = self._collection_ids(collection_ids)
        if ids:
            payload["collection_ids"] = ids
        logger.debug("Albert search: limit=%s collections=%s", payload["limit"], ids)

        response = requests.post(
            url=f"{settings.AI_BASE_URL}/search",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        self.__check_response(response)
        # Réponse : {"object": "list", "data": [{"method", "score", "chunk": {...}}, ...]}
        results = response.json().get("data") or []
        chunks = [self._normalize_search_result(result) for result in results]
        return [chunk for chunk in chunks if chunk]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        """Reorder ``documents`` by relevance to ``query`` via POST /v1/rerank.

        Albert follows the Cohere v2 rerank convention and answers
        ``{"results": [{"index": int, "relevance_score": float}, ...]}``.
        The returned list keeps that shape, filtered to valid indexes.
        """
        query = (query or "").strip()
        if not query or not documents:
            return []

        response = requests.post(
            url=f"{settings.AI_BASE_URL}/rerank",
            headers=self.headers,
            json={
                "model": settings.AI_RAG_RERANK_MODEL,
                "query": query[: settings.AI_QUERY_MAX_CHARS],
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            timeout=60,
        )
        self.__check_response(response)
        payload = response.json()
        results = payload.get("results")
        if results is None:
            # Certaines passerelles renvoient la liste sous "data".
            results = payload.get("data") or []
        return [
            result
            for result in results
            if isinstance(result.get("index"), int)
            and 0 <= result["index"] < len(documents)
        ]

    def call_ai_api(self, prompt, system_prompt: str | None = None, temperature=None):
        """Helper method to call the OpenAI API and process the response.

        ``system_prompt`` and ``temperature`` are optional so existing callers
        (summarizer, classifier) keep working unchanged.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": settings.AI_MODEL,
            "messages": messages,
#            "stream": False,
#            "n": 1,
        }
        if temperature is not None:
            data["temperature"] = temperature

        try:
            response = self.client.chat.completions.create(**data)
        except Exception:
            logger.exception("AI API call failed")
            raise

        if not response.choices:
            raise ValueError("AI response returned no choices")

        content = response.choices[0].message.content

        if not content:
            raise ValueError("AI response does not contain an answer")

        return content

    def upload_document(self, file_path: str) -> int:
        """Importe un fichier (PDF, TXT, HTML, MARKDOWN, max 20 Mo) dans la collection.

        L'API extrait le texte, le découpe en chunks, les vectorise puis les stocke.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        extension = os.path.splitext(file_path)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Format non accepté : '{extension or 'sans extension'}'. "
                f"Formats autorisés : {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"Fichier trop volumineux : {file_size / (1024 * 1024):.1f} Mo "
                f"(max {MAX_FILE_SIZE // (1024 * 1024)} Mo)"
            )

        # Type MIME déterminé depuis la liste blanche (prioritaire sur mimetypes,
        # qui peut renvoyer None ou une valeur inattendue selon la plateforme)
        mime_type = ALLOWED_EXTENSIONS[extension]

        with open(file_path, "rb") as f:
            response = requests.post(
                url=f"{settings.AI_BASE_URL}/documents",
                headers=self.headers,
                files={"file": (os.path.basename(file_path), f, mime_type)},
                data={"collection_id": str(settings.AI_PRIVATE_COLLECTION_ID)},
                timeout=300,
            )
        response.raise_for_status()
        return response.json()["id"]

    def get_private_collections(self) -> list:
        """Return the dictionary of private collections."""
        return settings.AI_PRIVATE_COLLECTION_ID
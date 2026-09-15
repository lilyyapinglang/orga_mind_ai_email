"""Service for AI-powered features using OpenAI-compatible API."""

import logging

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from openai import OpenAI

from core.ai.utils import is_ai_enabled

logger = logging.getLogger(__name__)


class AIService:
    """Service class for AI-related operations."""

    def __init__(self):
        """Ensure that the AI configuration is set properly."""
        if not is_ai_enabled():
            raise ImproperlyConfigured("AI configuration not set")
        self.client = OpenAI(
            base_url=settings.AI_BASE_URL,
            api_key=settings.AI_API_KEY,
            timeout=60,
            max_retries=1,
        )

    def call_ai_api(self, prompt, system_prompt=None):
        """Helper method to call the OpenAI API and process the response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "model": settings.AI_MODEL,
            "messages": messages,
            "stream": False,
            "n": 1,
        }

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

    def search(self, query):
        """Return relevant Albert RAG chunks from the configured collections."""
        try:
            collection_ids = [
                int(collection_id) for collection_id in settings.AI_RAG_COLLECTION_IDS
            ]
        except (TypeError, ValueError) as exc:
            raise ImproperlyConfigured(
                "AI_RAG_COLLECTION_IDS must contain integer collection IDs"
            ) from exc

        if not collection_ids:
            raise ImproperlyConfigured("AI_RAG_COLLECTION_IDS must not be empty")

        base_url = settings.AI_BASE_URL.rstrip("/")
        response = requests.post(
            f"{base_url}/search",
            headers={"Authorization": f"Bearer {settings.AI_API_KEY}"},
            json={
                "query": query,
                "collection_ids": collection_ids,
                "method": "hybrid",
                "limit": settings.AI_RAG_SEARCH_LIMIT,
            },
            timeout=60,
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        return [
            result["chunk"]["content"]
            for result in data
            if isinstance(result, dict)
            and isinstance(result.get("chunk"), dict)
            and isinstance(result["chunk"].get("content"), str)
        ]

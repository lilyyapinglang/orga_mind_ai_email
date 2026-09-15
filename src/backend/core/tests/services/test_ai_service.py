"""Tests for Albert RAG integration."""

from unittest.mock import Mock, patch

from django.test import override_settings
from django.core.exceptions import ImproperlyConfigured
import pytest

from core.api.viewsets.ai_draft import generate_ai_reply_body
from core.services.ai_service import AIService


@override_settings(
    AI_API_KEY="test-key",
    AI_BASE_URL="https://albert.example/v1/",
    AI_MODEL="albert-model",
    AI_RAG_COLLECTION_IDS=["150281", "150277"],
    AI_RAG_SEARCH_LIMIT=2,
)
@patch("core.services.ai_service.OpenAI")
@patch("core.services.ai_service.requests.post")
def test_search_uses_configured_albert_collections(mock_post, _mock_openai):
    """Searches stay scoped to the configured official collections."""
    response = Mock()
    response.json.return_value = {
        "data": [
            {"chunk": {"content": "Official guidance"}},
            {"chunk": {"content": "More guidance"}},
        ]
    }
    mock_post.return_value = response

    assert AIService().search("How do I renew my ID card?") == [
        "Official guidance",
        "More guidance",
    ]

    mock_post.assert_called_once_with(
        "https://albert.example/v1/search",
        headers={"Authorization": "Bearer test-key"},
        json={
            "query": "How do I renew my ID card?",
            "collection_ids": [150281, 150277],
            "method": "hybrid",
            "limit": 2,
        },
        timeout=60,
    )
    response.raise_for_status.assert_called_once_with()


@override_settings(
    AI_API_KEY="test-key",
    AI_BASE_URL="https://albert.example/v1",
    AI_MODEL="albert-model",
    AI_RAG_COLLECTION_IDS=[],
)
@patch("core.services.ai_service.OpenAI")
def test_search_rejects_unscoped_collection_search(_mock_openai):
    """An empty allowlist must not accidentally search every collection."""
    with pytest.raises(
        ImproperlyConfigured, match="AI_RAG_COLLECTION_IDS must not be empty"
    ):
        AIService().search("question")


@patch("core.api.viewsets.ai_draft.AIService")
def test_ai_reply_uses_retrieved_excerpts_in_the_completion(mock_service):
    """The completion receives retrieved guidance rather than email text alone."""
    message = Mock()
    message.get_as_text.return_value = "Citizen question"
    service = mock_service.return_value
    service.search.return_value = ["Official guidance"]
    service.call_ai_api.return_value = "Suggested reply"

    assert generate_ai_reply_body(message) == "Suggested reply"

    service.search.assert_called_once_with("Citizen question")
    prompt = service.call_ai_api.call_args.args[0]
    assert "Citizen question" in prompt
    assert "Official guidance" in prompt
    assert service.call_ai_api.call_args.kwargs["system_prompt"]

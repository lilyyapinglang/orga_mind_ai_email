"""Tests for the RAG evidence retrieval pipeline."""

# pylint: disable=redefined-outer-name

from unittest import mock

import pytest
import requests
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from core.services import ai_rag
from core.services.ai_rag import EvidenceStatus, retrieve_evidence

pytestmark = pytest.mark.django_db


def _chunk(content, chunk_id="c1", score=0.5):
    """Build a normalized search result as AIService.search_chunks returns it."""
    return {
        "content": content,
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "collection_id": "col-1",
        "document_name": "service-public.pdf",
        "search_score": score,
        "search_method": "semantic",
        "rerank_score": None,
    }


@pytest.fixture
def service():
    """Patch AIService as used by the pipeline."""
    with mock.patch.object(ai_rag, "AIService") as factory:
        yield factory.return_value


@override_settings(AI_RAG_SEARCH_CANDIDATES=25, AI_RAG_CONTEXT_LIMIT=3)
def test_retrieve_evidence_requests_a_broad_candidate_set(service):
    """A broad candidate set must be requested, not the small default limit."""
    service.search_chunks.return_value = [_chunk("a")]
    service.rerank.return_value = [{"index": 0, "relevance_score": 0.9}]

    retrieve_evidence("Comment renouveler ma carte d'identité ?")

    assert service.search_chunks.call_args.kwargs["limit"] == 25


@override_settings(AI_RAG_CONTEXT_LIMIT=2, AI_RAG_MIN_RELEVANCE_SCORE=0.3)
def test_retrieve_evidence_reorders_and_truncates_by_rerank_score(service):
    """Excerpts are returned in reranked order, truncated to the context limit."""
    service.search_chunks.return_value = [
        _chunk("first", chunk_id="c1"),
        _chunk("second", chunk_id="c2"),
        _chunk("third", chunk_id="c3"),
    ]
    service.rerank.return_value = [
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.71},
    ]

    evidence = retrieve_evidence("question")

    assert evidence.status == EvidenceStatus.OK
    assert evidence.reranked is True
    assert [chunk["chunk_id"] for chunk in evidence.chunks] == ["c3", "c1"]
    assert evidence.chunks[0]["rerank_score"] == 0.95


@override_settings(AI_RAG_MIN_RELEVANCE_SCORE=0.5)
def test_retrieve_evidence_rejects_weak_evidence_when_threshold_is_set(service):
    """Chunks scored below the threshold must not become context."""
    service.search_chunks.return_value = [_chunk("loosely related")]
    service.rerank.return_value = [{"index": 0, "relevance_score": 0.12}]

    evidence = retrieve_evidence("question")

    assert evidence.status == EvidenceStatus.NOT_RELEVANT
    assert evidence.has_evidence is False


@override_settings(AI_RAG_MIN_RELEVANCE_SCORE=0.0)
def test_default_threshold_keeps_low_scoring_chunks(service):
    """The default threshold must not silently discard the whole context."""
    service.search_chunks.return_value = [_chunk("a")]
    service.rerank.return_value = [{"index": 0, "relevance_score": 0.02}]

    evidence = retrieve_evidence("question")

    assert evidence.status == EvidenceStatus.OK
    assert len(evidence.chunks) == 1


def test_collection_ids_are_not_coerced_to_strings():
    """Albert rejects the wrong id type, so ids are passed through as-is."""
    from core.services.ai_service import AIService

    with override_settings(
        AI_COLLECTION_IDS=[150277, 139226], AI_PRIVATE_COLLECTION_ID=None
    ):
        assert AIService._collection_ids() == [150277, 139226]


def test_retrieve_evidence_without_results(service):
    """An empty search result is reported as such."""
    service.search_chunks.return_value = []

    assert retrieve_evidence("question").status == EvidenceStatus.NO_RESULTS


def test_retrieve_evidence_when_albert_is_unavailable(service):
    """A failing search degrades to 'unavailable' instead of raising."""
    service.search_chunks.side_effect = requests.RequestException("boom")

    assert retrieve_evidence("question").status == EvidenceStatus.UNAVAILABLE


def test_retrieve_evidence_when_albert_is_not_configured(service):
    """A missing API key is reported without raising."""
    service.search_chunks.side_effect = ImproperlyConfigured

    assert retrieve_evidence("question").status == EvidenceStatus.NOT_CONFIGURED


def test_retrieve_evidence_with_empty_query(service):
    """An empty query never reaches Albert (guaranteed 422)."""
    assert retrieve_evidence("   ").status == EvidenceStatus.EMPTY_QUERY
    service.search_chunks.assert_not_called()


@override_settings(AI_RAG_CONTEXT_LIMIT=2)
def test_rerank_failure_falls_back_to_search_order(service):
    """A reranker outage must not drop the whole RAG context."""
    service.search_chunks.return_value = [_chunk("a", "c1"), _chunk("b", "c2")]
    service.rerank.side_effect = requests.RequestException("boom")

    evidence = retrieve_evidence("question")

    assert evidence.status == EvidenceStatus.OK
    assert evidence.reranked is False
    assert len(evidence.chunks) == 2


@override_settings(AI_RAG_RERANK_MODEL="")
def test_reranking_can_be_disabled(service):
    """With no rerank model configured, search order is used directly."""
    service.search_chunks.return_value = [_chunk("a")]

    evidence = retrieve_evidence("question")

    assert evidence.reranked is False
    service.rerank.assert_not_called()


def test_sources_metadata_exposes_provenance(service):
    """Each retained excerpt keeps ids and scores for later review."""
    service.search_chunks.return_value = [_chunk("a")]
    service.rerank.return_value = [{"index": 0, "relevance_score": 0.9}]

    source = retrieve_evidence("question").sources_metadata()[0]

    assert source["collection_id"] == "col-1"
    assert source["document_id"] == "doc-1"
    assert source["chunk_id"] == "c1"
    assert source["rerank_score"] == 0.9
    assert source["excerpt"] == "a"

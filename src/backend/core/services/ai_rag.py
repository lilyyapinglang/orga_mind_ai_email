"""Evidence retrieval pipeline for AI-generated replies.

Flow: Albert ``/v1/search`` (broad candidate set) -> ``/v1/rerank`` (precision)
-> evidence gate. The gate is the important part: when no excerpt is relevant
enough, the caller must *not* produce a factual answer, because an ungrounded
answer to a citizen about deadlines, eligibility or fees is worse than no
answer at all.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from core.services.ai_service import AIService

logger = logging.getLogger(__name__)


class EvidenceStatus(str, Enum):
    """Outcome of the retrieval pipeline."""

    OK = "ok"
    NO_RESULTS = "no_results"
    NOT_RELEVANT = "not_relevant"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    EMPTY_QUERY = "empty_query"


@dataclass
class Evidence:
    """Excerpts retained for generation, plus why they were retained."""

    status: EvidenceStatus = EvidenceStatus.NO_RESULTS
    chunks: list[dict] = field(default_factory=list)
    candidates_count: int = 0
    reranked: bool = False

    @property
    def has_evidence(self) -> bool:
        """True when at least one excerpt passed the relevance gate."""
        return bool(self.chunks)

    def sources_metadata(self) -> list[dict]:
        """Reviewable provenance for each retained excerpt."""
        return [
            {
                "index": position,
                "collection_id": chunk.get("collection_id"),
                "document_id": chunk.get("document_id"),
                "document_name": chunk.get("document_name"),
                "chunk_id": chunk.get("chunk_id"),
                "search_score": chunk.get("search_score"),
                "rerank_score": chunk.get("rerank_score"),
                "excerpt": chunk.get("content"),
            }
            for position, chunk in enumerate(self.chunks, start=1)
        ]


def _apply_rerank(query: str, candidates: list[dict], top_n: int) -> tuple[list[dict], bool]:
    """Return ``(chunks, reranked)`` after the Albert reranking step.

    On any reranker failure we degrade to the raw search order rather than
    dropping the whole RAG context; the relevance gate below then simply has no
    rerank score to check.
    """
    if not settings.AI_RAG_RERANK_MODEL:
        return candidates[:top_n], False

    documents = [chunk["content"] for chunk in candidates]
    try:
        results = AIService().rerank(query, documents, top_n)
    except (requests.RequestException, ValueError, KeyError):
        logger.exception("Albert rerank failed; falling back to raw search order.")
        return candidates[:top_n], False

    if not results:
        return candidates[:top_n], False

    ranked = []
    for result in results:
        chunk = dict(candidates[result["index"]])
        chunk["rerank_score"] = result.get("relevance_score")
        ranked.append(chunk)
    return ranked[:top_n], True


def _passes_relevance_gate(chunk: dict, reranked: bool) -> bool:
    """Keep an excerpt only if the reranker scored it above the threshold.

    Albert documents ``score_threshold`` as semantic-search only, so we never
    threshold the raw hybrid search score — only the reranker's own relevance
    score, which is comparable across queries.
    """
    if not reranked:
        return True
    score = chunk.get("rerank_score")
    if score is None:
        return True
    return float(score) >= settings.AI_RAG_MIN_RELEVANCE_SCORE


def retrieve_evidence(query: str) -> Evidence:
    """Retrieve, rerank and filter the excerpts backing a reply."""
    query = (query or "").strip()
    if not query:
        return Evidence(status=EvidenceStatus.EMPTY_QUERY)

    try:
        candidates = AIService().search_chunks(
            query, limit=settings.AI_RAG_SEARCH_CANDIDATES
        )
    except ImproperlyConfigured:
        logger.warning("Albert API not configured; no evidence retrieved.")
        return Evidence(status=EvidenceStatus.NOT_CONFIGURED)
    except (requests.RequestException, ValueError, KeyError):
        logger.exception("Albert search failed; no evidence retrieved.")
        return Evidence(status=EvidenceStatus.UNAVAILABLE)

    if not candidates:
        logger.info("Albert search returned no chunk for query: %s", query[:200])
        return Evidence(status=EvidenceStatus.NO_RESULTS)

    ranked, reranked = _apply_rerank(
        query, candidates, settings.AI_RAG_CONTEXT_LIMIT
    )
    kept = [chunk for chunk in ranked if _passes_relevance_gate(chunk, reranked)]

    if not kept:
        logger.info(
            "Albert rerank kept no chunk above %.2f (%d candidates) for query: %s",
            settings.AI_RAG_MIN_RELEVANCE_SCORE,
            len(candidates),
            query[:200],
        )
        return Evidence(
            status=EvidenceStatus.NOT_RELEVANT,
            candidates_count=len(candidates),
            reranked=reranked,
        )

    logger.info(
        "Albert RAG: %d candidates -> %d excerpts (reranked=%s)",
        len(candidates),
        len(kept),
        reranked,
    )
    return Evidence(
        status=EvidenceStatus.OK,
        chunks=kept,
        candidates_count=len(candidates),
        reranked=reranked,
    )

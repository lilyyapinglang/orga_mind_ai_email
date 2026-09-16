"""Tests for the grounded AI draft generation contract."""

# pylint: disable=redefined-outer-name

from unittest import mock

import pytest
from django.test import override_settings

from core.api.viewsets import ai_draft
from core.api.viewsets.ai_draft import (
    SAFE_DRAFT_BODY,
    _parse_structured_reply,
    _validate_structured_reply,
    generate_ai_reply_body_with_rag,
)
from core.services.ai_rag import Evidence, EvidenceStatus

pytestmark = pytest.mark.django_db


def _chunk(content="extrait officiel", chunk_id="c1"):
    return {
        "content": content,
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "collection_id": "col-1",
        "document_name": "service-public.pdf",
        "search_score": 0.5,
        "rerank_score": 0.9,
    }


@pytest.fixture
def message():
    """A minimal stand-in for the source message."""
    source = mock.Mock()
    source.id = "11111111-1111-1111-1111-111111111111"
    source.thread_id = None
    source.get_as_text.return_value = "Bonjour, quel est le délai de traitement ?"
    return source


@pytest.fixture
def ai_service():
    """Patch the AI service used for generation."""
    with mock.patch.object(ai_draft, "AIService") as factory:
        yield factory.return_value


# --- structured output parsing ------------------------------------------------


def test_parse_plain_json():
    """A bare JSON object is parsed."""
    parsed = _parse_structured_reply('{"body": "Bonjour", "source_indexes": [1]}')
    assert parsed["body"] == "Bonjour"


def test_parse_json_inside_markdown_fence():
    """A fenced JSON object is parsed despite the contract forbidding fences."""
    parsed = _parse_structured_reply('```json\n{"body": "Bonjour"}\n```')
    assert parsed["body"] == "Bonjour"


def test_parse_json_with_surrounding_prose():
    """A JSON object wrapped in chatter is still recovered."""
    parsed = _parse_structured_reply('Voici :\n{"body": "Bonjour"}\nMerci.')
    assert parsed["body"] == "Bonjour"


def test_parse_non_json_returns_none():
    """Free text is not mistaken for a structured answer."""
    assert _parse_structured_reply("Madame, Monsieur, ...") is None


def test_validate_drops_out_of_range_source_indexes():
    """Hallucinated excerpt numbers are discarded."""
    reply = _validate_structured_reply(
        {"body": "ok", "source_indexes": [1, 7, "2", None]}, excerpt_count=3
    )
    assert reply.source_indexes == [1, 2]


def test_validate_flags_unanswered_points_for_review():
    """Any uncovered point forces human review."""
    reply = _validate_structured_reply(
        {
            "body": "ok",
            "needs_human_review": False,
            "unanswered_points": ["montant exact de l'aide"],
            "source_indexes": [1],
        },
        excerpt_count=2,
    )
    assert reply.needs_human_review is True


def test_validate_flags_answer_citing_no_excerpt():
    """A grounded answer that cites nothing is treated as unsupported."""
    reply = _validate_structured_reply(
        {"body": "ok", "needs_human_review": False, "source_indexes": []},
        excerpt_count=3,
    )
    assert reply.needs_human_review is True


# --- generation ---------------------------------------------------------------


@override_settings(AI_RAG_REQUIRE_EVIDENCE=True)
def test_no_evidence_generates_a_fact_free_acknowledgement(message, ai_service):
    """Without evidence, the reply acknowledges but states no fact."""
    ai_service.call_ai_api.return_value = "Madame, Monsieur,\n\nNous vérifions."
    with mock.patch.object(
        ai_draft.ai_rag,
        "retrieve_evidence",
        return_value=Evidence(status=EvidenceStatus.NO_RESULTS),
    ):
        reply = generate_ai_reply_body_with_rag(message)

    assert reply.body == "Madame, Monsieur,\n\nNous vérifions."
    assert reply.needs_human_review is True
    assert reply.grounded is False
    assert reply.unanswered_points
    prompt = ai_service.call_ai_api.call_args.args[0]
    assert "No official source excerpt is available" in prompt
    assert "[Extrait" not in prompt


@override_settings(AI_RAG_REQUIRE_EVIDENCE=True)
def test_no_evidence_falls_back_to_template_when_generation_fails(
    message, ai_service
):
    """A model outage on the no-evidence path still yields a sendable draft."""
    ai_service.call_ai_api.side_effect = RuntimeError("model down")
    with mock.patch.object(
        ai_draft.ai_rag,
        "retrieve_evidence",
        return_value=Evidence(status=EvidenceStatus.NOT_RELEVANT),
    ):
        reply = generate_ai_reply_body_with_rag(message)

    assert reply.body == SAFE_DRAFT_BODY
    assert reply.evidence_status == EvidenceStatus.NOT_RELEVANT.value


def test_search_query_drops_politeness_boilerplate():
    """Salutations and closings must not dilute the retrieval query."""
    cleaned = ai_draft._clean_search_query(
        "Bonjour,\n\nJe n'ai pas reçu mon allocation après trois semaines "
        "et je voudrais savoir si ce délai est normal.\n\nMerci d'avance.\n"
        "Cordialement,\nAntoine"
    )
    assert "Bonjour" not in cleaned
    assert "Cordialement" not in cleaned
    assert "allocation" in cleaned


def test_search_query_keeps_original_when_cleanup_removes_everything():
    """The heuristic must never produce a near-empty query."""
    assert ai_draft._clean_search_query("Bonjour,\nCordialement,") != ""


@override_settings(AI_RAG_REQUIRE_EVIDENCE=False)
def test_ungrounded_reply_is_flagged_when_evidence_not_required(message, ai_service):
    """Opting out of the gate still marks the draft as unverified."""
    ai_service.call_ai_api.return_value = "Madame, Monsieur, ..."
    with mock.patch.object(
        ai_draft.ai_rag,
        "retrieve_evidence",
        return_value=Evidence(status=EvidenceStatus.NO_RESULTS),
    ):
        reply = generate_ai_reply_body_with_rag(message)

    assert reply.grounded is False
    assert reply.needs_human_review is True


def test_grounded_reply_carries_excerpts_and_sources(message, ai_service):
    """A grounded answer keeps the excerpts in the prompt and in the metadata."""
    evidence = Evidence(
        status=EvidenceStatus.OK,
        chunks=[_chunk("Le délai est de deux mois.")],
        candidates_count=25,
        reranked=True,
    )
    ai_service.call_ai_api.return_value = (
        '{"body": "Madame, Monsieur,\\n\\nLe délai est de deux mois.", '
        '"needs_human_review": false, "unanswered_points": [], '
        '"source_indexes": [1]}'
    )
    with mock.patch.object(
        ai_draft.ai_rag, "retrieve_evidence", return_value=evidence
    ):
        reply = generate_ai_reply_body_with_rag(message)

    prompt = ai_service.call_ai_api.call_args.args[0]
    assert "Le délai est de deux mois." in prompt
    assert "[Extrait 1]" in prompt
    assert reply.grounded is True
    assert reply.needs_human_review is False
    assert reply.source_indexes == [1]

    metadata = reply.metadata()
    assert metadata["sources"][0]["used"] is True
    assert metadata["sources"][0]["chunk_id"] == "c1"


def test_unstructured_model_answer_is_flagged_for_review(message, ai_service):
    """If the model ignores the JSON contract, the draft needs review."""
    evidence = Evidence(status=EvidenceStatus.OK, chunks=[_chunk()])
    ai_service.call_ai_api.return_value = "Madame, Monsieur, voici la réponse."
    with mock.patch.object(
        ai_draft.ai_rag, "retrieve_evidence", return_value=evidence
    ):
        reply = generate_ai_reply_body_with_rag(message)

    assert reply.body == "Madame, Monsieur, voici la réponse."
    assert reply.needs_human_review is True


def test_empty_body_raises(message, ai_service):
    """An empty body is an error, never an empty draft sent to a citizen."""
    evidence = Evidence(status=EvidenceStatus.OK, chunks=[_chunk()])
    ai_service.call_ai_api.return_value = '{"body": "   "}'
    with mock.patch.object(
        ai_draft.ai_rag, "retrieve_evidence", return_value=evidence
    ):
        with pytest.raises(ValueError):
            generate_ai_reply_body_with_rag(message)

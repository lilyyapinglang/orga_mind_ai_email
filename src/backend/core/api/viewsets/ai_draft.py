"""API view for generating an AI reply and saving it as a draft."""

import json
import logging
import os

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Exists, OuterRef

import rest_framework as drf
import requests
from drf_spectacular.utils import OpenApiExample, extend_schema, inline_serializer
from rest_framework import serializers as drf_serializers
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from core import enums, models
from core.mda.draft import create_draft
from core.services.ai_service import AIService

from .. import permissions, serializers

logger = logging.getLogger(__name__)

THREAD_CONTEXT_MAX_MESSAGES = 8
THREAD_CONTEXT_MAX_CHARS_PER_MESSAGE = 2000


def build_rag_context(question: str, results: list[dict]) -> str:
    """Build a provenance-preserving RAG context block.

    The source identifiers come directly from the Albert /search response.
    No source metadata is inferred or reconstructed.
    """
    excerpts = []

    for index, result in enumerate(results, start=1):
        chunk = result.get("chunk") or {}
        metadata = chunk.get("metadata") or {}

        title = metadata.get("title") or "Source officielle"
        url = metadata.get("url") or ""
        chunk_id = metadata.get("_chunk_id") or chunk.get("id")
        document_id = metadata.get("_doc_id") or chunk.get("document_id")
        score = result.get("score")
        content = chunk.get("content")

        if not content:
            continue

        source_lines = [
            f"[Source {index}]",
            f"Titre: {title}",
        ]

        if url:
            source_lines.append(f"URL: {url}")

        if chunk_id is not None:
            source_lines.append(f"Chunk: {chunk_id}")

        if document_id is not None:
            source_lines.append(f"Document: {document_id}")

        if score is not None:
            source_lines.append(f"Score de recherche: {score}")

        source_lines.append(f"Contenu:\n{content}")

        excerpts.append("\n".join(source_lines))

    if not excerpts:
        return (
            "Aucune source officielle exploitable n'a été retrouvée.\n"
            f"\n[Question]\n{question}"
        )

    return (
        "Les sources ci-dessous sont les seules sources officielles "
        "fournies pour établir les faits administratifs de la réponse.\n"
        "Les identifiants [Source X] servent uniquement à identifier les "
        "sources et ne doivent pas apparaître dans la réponse destinée au citoyen.\n"
        f"\n[Question]\n{question}"
        f"\n\n[Sources officielles]\n\n"
        + "\n\n".join(excerpts)
    )


def _clip_text(text: str, max_chars: int) -> str:
    """Keep prompt chunks bounded while preserving the beginning of each message."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n[truncated]"


def _build_thread_context(message: models.Message) -> str:
    """Return a compact chronological transcript for the source message thread."""
    if not message.thread_id:
        return message.get_as_text()

    thread_messages = list(
        models.Message.objects.select_related("sender")
        .prefetch_related("recipients__contact")
        .filter(thread_id=message.thread_id, is_draft=False)
        .order_by("-created_at", "-id")[:THREAD_CONTEXT_MAX_MESSAGES]
    )
    thread_messages.reverse()

    if not thread_messages:
        return message.get_as_text()

    entries = []
    for index, thread_message in enumerate(thread_messages, start=1):
        marker = (
            "source message" if thread_message.id == message.id else "thread message"
        )
        message_text = _clip_text(
            thread_message.get_as_text(), THREAD_CONTEXT_MAX_CHARS_PER_MESSAGE
        )
        entries.append(f"[{marker} {index}]\n{message_text}")
    return "\n\n".join(entries)


def _rag_search_query(
    message: models.Message, current_draft_text: str | None
) -> str:
    """Build the Albert search query from the citizen's email thread only.

    The current agent draft is intentionally excluded from retrieval because
    it represents an agent hypothesis/intent, not authoritative evidence.
    Including it in the search query could bias retrieval toward supporting
    an already-written claim.
    """
    del current_draft_text  # Retrieval must not depend on the agent draft.
    return (_build_thread_context(message) or "").strip()


class ServiceUnavailable(drf.exceptions.APIException):
    """503 response for unavailable upstream AI service."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = "service_unavailable"


def _reply_subject(subject: str | None) -> str:
    """Return a reply subject without stacking repeated Re: prefixes."""
    subject = subject or ""
    if subject.lower().startswith("re:"):
        return subject
    return f"Re: {subject}" if subject else "Re:"


def _blocknote_paragraphs(text: str) -> str:
    """Serialize plain AI text into the draft editor's BlockNote JSON shape."""
    paragraphs = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraphs.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": paragraph, "styles": {}}],
            }
        )
    if not paragraphs:
        paragraphs = [{"type": "paragraph", "content": ""}]
    return json.dumps(paragraphs)


def _build_prompt(
    message: models.Message,
    current_draft_text: str | None = None,
    rag_context: str | None = None,
) -> str:
    """Build the prompt used to generate a citizen-facing reply."""

    draft_instruction = ""
    if current_draft_text and current_draft_text.strip():
        draft_instruction = (
            "Agent draft or intent:\n"
            f"{current_draft_text.strip()}\n\n"
            "Use this draft only to understand the agent's intended "
            "direction, wording, or requested answer.\n"
            "The agent draft is NOT an authoritative source of facts.\n"
            "Do not use a factual claim from the agent draft unless that "
            "claim is supported by the citizen's email or by an official "
            "source provided below.\n"
            "If the agent draft conflicts with the citizen's email or the "
            "official sources, do not use the conflicting claim.\n\n"
        )

    if rag_context:
        source_instruction = (
            "Official reference material:\n"
            f"{rag_context}\n\n"
            "Rules for using the official reference material:\n"
            "- Use official sources as the authority for administrative, "
            "legal, procedural, eligibility, deadline, and timing facts.\n"
            "- Every administrative factual claim in the reply must be "
            "supported by the provided official sources.\n"
            "- Do not infer a deadline, processing time, eligibility "
            "condition, procedure, or obligation from incomplete information.\n"
            "- Do not turn a maximum duration into a normal processing time.\n"
            "- Do not turn an exception into a general rule.\n"
            "- Do not turn an example into a general rule.\n"
            "- Do not combine separate statements to manufacture a deadline "
            "or other factual rule that the source does not explicitly establish.\n"
            "- If the sources do not establish the answer, say so clearly "
            "and ask for the missing information when appropriate.\n"
            "- Do not use general knowledge or model knowledge to fill gaps "
            "in the official sources.\n"
            "- Do not mention the internal source labels or retrieval process "
            "to the citizen.\n\n"
        )
    else:
        source_instruction = (
            "No official reference material is currently available.\n"
            "Do not use model knowledge to invent or assert administrative, "
            "legal, procedural, eligibility, deadline, or timing facts.\n"
            "If the citizen's request requires such information, explain "
            "that the available information is insufficient and ask for the "
            "missing information or direct the citizen to the appropriate "
            "official service when that can be stated without inventing "
            "specific details.\n\n"
        )

    return (
        "You are helping a government agent draft a reply to a citizen.\n\n"
        "Write only the reply body. Do not include a subject line.\n\n"
        "Authority hierarchy:\n"
        "1. The citizen email/thread is authoritative for facts about the "
        "citizen's own situation, dates, names, amounts, and case details.\n"
        "2. The provided official sources are authoritative for "
        "administrative, legal, procedural, eligibility, deadline, and "
        "timing information.\n"
        "3. The agent draft is only an expression of intent. It is not "
        "evidence and must never be treated as authoritative by itself.\n\n"
        "Requirements:\n"
        "- Be concise. Keep the reply short, ideally under 150 words.\n"
        "- Use a formal and professional tone, as expected from a public "
        "administration or government office.\n"
        "- Structure the reply as a professional email: a formal salutation "
        "(for example 'Madame, Monsieur'), a short body of one to three "
        "paragraphs, and a formal closing formula (for example 'Je vous prie "
        "d'agréer, Madame, Monsieur, l'expression de mes salutations "
        "distinguées.') followed by the signature placeholder of the "
        "administration.\n"
        "- Reply in the same language as the citizen's email.\n"
        "- Do not use Markdown formatting.\n"
        "- Do not invent facts, promises, dates, amounts, deadlines, "
        "processing times, procedures, or case details.\n"
        "- If required information is missing, say what cannot be determined "
        "and ask for it briefly.\n"
        "- Consider the full email thread in chronological order.\n"
        "- Reply to the latest/source citizen message while preserving "
        "relevant facts, commitments, answers, and unresolved requests from "
        "earlier messages in the same thread.\n\n"
        f"Email thread:\n{_build_thread_context(message)}\n\n"
        f"{source_instruction}"
        f"{draft_instruction}"
        "Draft reply:\n\n"
    )


def generate_ai_reply_body(
        message: models.Message, current_draft_text: str | None = None
) -> str:
    """Generate the reply body for a message using the configured AI service.

    Flow: search official-doc chunks via Albert API, then let the AI service
    draft the reply with that context. If the RAG step fails or is not
    configured, fall back to a plain AI reply without context.
    """
    try:
        return AIService().call_ai_api(_build_prompt(message, current_draft_text))
    except Exception:
        logger.exception("AI service failed without RAG context, re-raising")
        raise

def generate_ai_reply_body_with_rag(
    message: models.Message, current_draft_text: str | None = None
) -> str:
    """Generate a reply using official evidence retrieved from Albert.

    If official evidence is unavailable, generation remains possible but the
    prompt explicitly prevents the model from filling administrative gaps
    from its own knowledge.
    """
    query = _rag_search_query(message, current_draft_text)
    results: list[dict] = []

    if query:
        try:
            results = AIService().search_chunks(query)
            logger.info(
                "Albert RAG: %d chunks retrieved for query: %s",
                len(results),
                query[:200],
            )
        except ImproperlyConfigured:
            logger.warning(
                "Albert API key not configured; generating without official "
                "reference context."
            )
        except requests.RequestException:
            logger.exception(
                "Albert RAG search failed; generating without official "
                "reference context."
            )

    context_block = build_rag_context(query, results) if results else None

    prompt = _build_prompt(
        message,
        current_draft_text=current_draft_text,
        rag_context=context_block,
    )

    return AIService().call_ai_api(prompt)


def generate_preview_reply_body(message: models.Message) -> str:
    """Generate a local preview reply while the AI/MCP pipeline is not wired yet."""
    sender_name = message.sender.name or message.sender.email or "there"
    return (
        f"Hello {sender_name},\n\n"
        "Thank you for your message. We have received your request and will review "
        "the information you provided.\n\n"
        "This is a preview draft generated before the AI and official document "
        "retrieval pipeline is connected. Once that pipeline is available, this "
        "draft will be replaced by an answer based on the citizen email and the "
        "retrieved official documentation.\n\n"
        "TODO TODO!!!\n\n"
        "Best regards,"
    )


@extend_schema(tags=["messages"])
class AIDraftView(APIView):
    """Generate an AI reply to a message and store the result as a draft."""

    permission_classes = [permissions.IsAuthenticated]

    @staticmethod
    def _get_source_message(user, message_id):
        """Return the non-draft source message if the user can read its thread."""
        try:
            return (
                models.Message.objects.select_related("sender", "thread")
                .prefetch_related("recipients__contact")
                .filter(
                    Exists(
                        models.ThreadAccess.objects.filter(
                            mailbox__accesses__user=user,
                            thread=OuterRef("thread_id"),
                        )
                    )
                )
                .get(id=message_id, is_draft=False)
            )
        except models.Message.DoesNotExist as exc:
            raise drf.exceptions.NotFound(
                "Message not found, is a draft, or access denied."
            ) from exc

    @staticmethod
    def _get_sender_mailbox(user, sender_id, thread):
        """Return the mailbox the user may use to create this reply draft."""
        if not sender_id:
            raise drf.exceptions.ValidationError({"senderId": "This field is required."})

        try:
            mailbox = models.Mailbox.objects.filter(
                id=sender_id,
                accesses__user=user,
                accesses__role__in=enums.MAILBOX_ROLES_CAN_EDIT,
            ).first()
        except (ValueError, TypeError):
            mailbox = None

        if mailbox is None:
            raise drf.exceptions.PermissionDenied(
                "You do not have permission to draft as this mailbox."
            )

        can_reply_in_thread = models.ThreadAccess.objects.filter(
            thread=thread,
            mailbox=mailbox,
            role=enums.ThreadAccessRoleChoices.EDITOR,
        ).exists()
        if not can_reply_in_thread:
            raise drf.exceptions.PermissionDenied(
                "This mailbox cannot create a draft in the source message thread."
            )
        return mailbox

    @extend_schema(
        summary="Generate an AI reply draft",
        request=inline_serializer(
            name="AIDraftRequest",
            fields={
                "senderId": drf_serializers.UUIDField(
                    required=True,
                    help_text="Mailbox ID to use as the draft sender.",
                ),
                "currentDraftText": drf_serializers.CharField(
                    required=False,
                    allow_blank=True,
                    help_text=(
                            "Current composer draft or short intent to expand into the "
                            "AI reply."
                    ),
                ),
            },
        ),
        responses={
            201: serializers.MessageSerializer,
            400: OpenApiExample(
                "Validation Error",
                value={"senderId": "This field is required."},
            ),
            403: OpenApiExample(
                "Permission Error",
                value={"detail": "You do not have permission to draft as this mailbox."},
            ),
            404: OpenApiExample(
                "Not Found",
                value={"detail": "Message not found, is a draft, or access denied."},
            ),
            503: OpenApiExample(
                "AI Not Configured",
                value={"detail": "AI service is not configured."},
            ),
        },
        description=(
                "Generate a citizen-facing reply with the configured AI service and "
                "save it as a draft reply to the source message. The citizen email "
                "address is taken from the source message sender."
        ),
    )
    def post(self, request, message_id):
        """Generate an AI response to ``message_id`` and create a reply draft."""
        source_message = self._get_source_message(request.user, message_id)
        sender_mailbox = self._get_sender_mailbox(
            request.user,
            request.data.get("senderId"),
            source_message.thread,
        )
        current_draft_text = request.data.get("currentDraftText")

        if settings.AI_DRAFT_PREVIEW_ONLY:
            ai_reply = generate_preview_reply_body(source_message)
        else:
            try:
                ai_reply = generate_ai_reply_body_with_rag(
                    source_message, current_draft_text
                )
            except ImproperlyConfigured:
                logger.info(
                    "AI service is not configured; creating preview AI draft for message %s",
                    message_id,
                )
                ai_reply = generate_preview_reply_body(source_message)
            except Exception as exc:
                logger.exception("Failed to generate AI draft for message %s", message_id)
                raise ServiceUnavailable("Failed to generate AI draft.") from exc

        draft = create_draft(
            mailbox=sender_mailbox,
            subject=_reply_subject(source_message.subject),
            draft_body=_blocknote_paragraphs(ai_reply),
            parent_id=str(source_message.id),
            to_emails=[source_message.sender.email],
            cc_emails=[],
            bcc_emails=[],
            attachments=[],
            user=request.user,
        )

        draft = models.Message.objects.with_read_state(sender_mailbox.id).get(
            id=draft.id
        )
        return Response(
            serializers.MessageSerializer(draft, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )

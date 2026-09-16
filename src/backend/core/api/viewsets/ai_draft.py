"""API view for generating an AI reply and saving it as a draft."""

import dataclasses
import json
import logging
import os
import re

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Exists, OuterRef

import rest_framework as drf
from drf_spectacular.utils import OpenApiExample, extend_schema, inline_serializer
from rest_framework import serializers as drf_serializers
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from core import enums, models
from core.mda.draft import create_draft
from core.services import ai_rag
from core.services.ai_service import AIService

from .. import permissions, serializers

logger = logging.getLogger(__name__)

THREAD_CONTEXT_MAX_MESSAGES = 8
THREAD_CONTEXT_MAX_CHARS_PER_MESSAGE = 2000


def build_rag_context(question: str, chunks: list[dict]) -> str:
    """Build the RAG context block from the retained excerpts.

    Excerpts are numbered so the model can reference them in ``source_indexes``
    and so an agent can map a sentence back to an official source.
    """
    extraits = "\n\n".join(
        f"[Extrait {i}]\n{chunk['content']}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return (
        "Réponds uniquement en t'appuyant sur les extraits fournis.\n"
        f"\n[Question]\n{question}\n"
        f"\n[Extraits]\n{extraits}"
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


_QUERY_BOILERPLATE = re.compile(
    r"^\s*("
    r"bonjour|bonsoir|madame|monsieur|madame,\s*monsieur|cher|chère"
    r"|cordialement|sincèrement|merci d'avance|merci beaucoup|merci"
    r"|je vous remercie|dans l'attente|bien à vous|salutations"
    r"|je vous prie d'agréer|veuillez agréer"
    r")\b.*$",
    re.IGNORECASE,
)


def _clean_search_query(text: str) -> str:
    """Drop salutations, closings and signatures from the retrieval query.

    Politeness boilerplate is a large share of a short citizen email and it
    matches nothing in an administrative corpus, so it dilutes both the vector
    search and the reranker. This only affects retrieval: the generation prompt
    still receives the untouched thread.
    """
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or _QUERY_BOILERPLATE.match(stripped):
            continue
        lines.append(stripped)
    cleaned = " ".join(lines).strip()
    # If the cleanup ate almost everything, the heuristic misfired: keep the
    # original rather than sending a near-empty query.
    return cleaned if len(cleaned) >= 25 else (text or "").strip()


def _rag_search_query(message: models.Message, current_draft_text: str | None) -> str:
    """Build the search query: email thread first, agent draft as fallback.

    The email thread is the actual question; the agent draft is only an
    intent and may be empty — it must never be sent as-is to /v1/search.
    """
    citizen_text = _clean_search_query(_build_thread_context(message) or "")
    draft_text = (current_draft_text or "").strip()
    if citizen_text and draft_text:
        return f"{citizen_text}\n\nAgent draft intent:\n{draft_text}"
    return citizen_text or draft_text


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


def _build_prompt(message: models.Message, current_draft_text: str | None = None) -> str:
    """Build the prompt used to generate a citizen-facing reply."""
    draft_instruction = ""
    if current_draft_text and current_draft_text.strip():
        draft_instruction = (
            "Agent draft or intent to preserve and expand:\n"
            f"{current_draft_text.strip()}\n\n"
            "Use this draft as the main intent of the reply, even if it is very "
            "short, for example yes/no/a day of the week. Expand it into a "
            "complete formal reply suitable for a public administration or "
            "government office.\n"
            "However, if the draft contradicts any information contained in "
            "the citizen's email (dates, amounts, names, case details, or any "
            "other fact), ignore the contradicting part of the draft and rely "
            "solely on the citizen's email. Never include a statement from "
            "the draft that conflicts with the information in the email.\n\n"
        )

    return (
        "You are helping a government agent draft a reply to a citizen.\n\n"
        "Write only the reply body. Do not include a subject line.\n\n"
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
        "- You only can reply in the language of the citizen's email.\n"
        "- Make sure your response is in the same language as the citizen's email.\n"
        "- Do not use any Markdown formatting: no headings, no bold, no "
        "italic, no bullet lists, no asterisks. Plain text only.\n"
        "- Do not invent facts, promises, dates, or case details that are "
        "not in the email thread. If information is missing, ask for it briefly.\n"
        "- Consider the full email thread below in chronological order. Reply "
        "to the latest/source citizen message, while preserving relevant "
        "facts, commitments, answers, and unresolved requests from earlier "
        "messages in the same thread.\n\n"
        f"Email thread:\n{_build_thread_context(message)}\n\n"
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


GROUNDING_CONTRACT = (
    "Grounding rules (these override every other instruction):\n"
    "- Every factual statement in the reply must be supported by one of the "
    "numbered excerpts. A fact with no supporting excerpt must not be written.\n"
    "- Never combine two unrelated excerpts to build a new rule, threshold or "
    "procedure that neither excerpt states on its own.\n"
    "- Never infer eligibility, deadlines, amounts, fees, processing times or "
    "legal consequences. State only what an excerpt says explicitly.\n"
    "- Facts about this citizen's own case (dates, amounts, names, reference "
    "numbers) come from the email thread, not from the excerpts.\n"
    "- Do not cite excerpt numbers, URLs or document names in the reply body: "
    "the agent reviews sources separately.\n"
    "- List in 'unanswered_points' every part of the citizen's request that the "
    "excerpts do not cover, and set 'needs_human_review' to true whenever such "
    "a part exists or you are unsure.\n"
)

STRUCTURED_OUTPUT_CONTRACT = (
    "Return ONLY a JSON object, with no Markdown fence and no text around it, "
    "with exactly these keys:\n"
    '{"body": "the reply body as plain text, \\n for line breaks", '
    '"needs_human_review": true or false, '
    '"unanswered_points": ["short description of each uncovered point"], '
    '"source_indexes": [numbers of the excerpts actually used]}\n'
)

# Contract used when no official excerpt supports the answer. The reply is
# still written for this specific citizen, but it may not contain a single
# administrative fact — only what the citizen themselves wrote.
NO_EVIDENCE_CONTRACT = (
    "No official source excerpt is available for this request.\n"
    "Write an acknowledgement reply under these absolute rules:\n"
    "- State NO administrative fact: no delay, no processing time, no amount, "
    "no eligibility condition, no deadline, no fee, no procedure step, no "
    "legal reference. Not even ones you believe are common knowledge.\n"
    "- You may restate what the citizen wrote (their situation, their dates, "
    "their request) to show the message was understood.\n"
    "- Say that the request is being checked by the department and that a "
    "reply will follow.\n"
    "- If identifying information would obviously speed this up (file number, "
    "reference number), ask for it in one short sentence.\n"
    "- Keep the formal salutation, tone and closing formula.\n"
)

# Last-resort reply, used only if the no-evidence generation itself fails.
SAFE_DRAFT_BODY = (
    "Madame, Monsieur,\n\n"
    "Nous accusons réception de votre message.\n\n"
    "Votre demande nécessite une vérification par notre service. Nous revenons "
    "vers vous dès que possible.\n\n"
    "Je vous prie d'agréer, Madame, Monsieur, l'expression de mes salutations "
    "distinguées.\n\n"
    "[Signature]"
)


@dataclasses.dataclass
class GeneratedReply:
    """A generated draft body plus everything an agent needs to review it."""

    body: str
    needs_human_review: bool = False
    unanswered_points: list = dataclasses.field(default_factory=list)
    source_indexes: list = dataclasses.field(default_factory=list)
    evidence_status: str = ai_rag.EvidenceStatus.NO_RESULTS.value
    sources: list = dataclasses.field(default_factory=list)
    grounded: bool = False

    def metadata(self) -> dict:
        """Reviewable metadata returned alongside the created draft."""
        used = set(self.source_indexes)
        return {
            "needsHumanReview": self.needs_human_review,
            "unansweredPoints": self.unanswered_points,
            "evidenceStatus": self.evidence_status,
            "grounded": self.grounded,
            "model": settings.AI_MODEL,
            "promptVersion": settings.AI_PROMPT_VERSION,
            "rerankModel": settings.AI_RAG_RERANK_MODEL or None,
            "sources": [
                {**source, "used": source["index"] in used} for source in self.sources
            ],
        }


def _parse_structured_reply(raw: str) -> dict | None:
    """Parse the model's JSON answer, tolerating fences and surrounding prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("body"), str):
            return parsed
    return None


def _validate_structured_reply(parsed: dict, excerpt_count: int) -> GeneratedReply:
    """Coerce the parsed payload into a trustworthy GeneratedReply."""
    body = parsed["body"].strip()

    unanswered = parsed.get("unanswered_points") or []
    if not isinstance(unanswered, list):
        unanswered = [str(unanswered)]
    unanswered = [str(point).strip() for point in unanswered if str(point).strip()]

    raw_indexes = parsed.get("source_indexes") or []
    if not isinstance(raw_indexes, list):
        raw_indexes = [raw_indexes]
    indexes = sorted(
        {
            int(index)
            for index in raw_indexes
            if isinstance(index, (int, float, str))
            and str(index).strip().lstrip("-").isdigit()
            and 1 <= int(index) <= excerpt_count
        }
    )

    needs_review = bool(parsed.get("needs_human_review")) or bool(unanswered)
    # A grounded answer that cites no excerpt at all is, by our own contract,
    # unsupported: flag it rather than presenting it as verified.
    if excerpt_count and not indexes:
        needs_review = True

    return GeneratedReply(
        body=body,
        needs_human_review=needs_review,
        unanswered_points=unanswered,
        source_indexes=indexes,
    )


def _build_grounded_prompt(
        message: models.Message,
        current_draft_text: str | None,
        context_block: str,
) -> str:
    """Add the excerpts, the grounding rules and the output contract."""
    prompt = _build_prompt(message, current_draft_text)
    return prompt.replace(
        "Draft reply:\n\n",
        f"Official reference excerpts to rely on:\n\n{context_block}\n\n"
        f"{GROUNDING_CONTRACT}\n"
        "Conflict rule:\n"
        "When the agent draft conflicts with these official excerpts, "
        "ignore the draft and answer according to the excerpts. Do not "
        "mention the conflict to the citizen unless clarification is "
        "needed.\n\n"
        f"{STRUCTURED_OUTPUT_CONTRACT}\n"
        "Draft reply:\n\n",
    )


def _generate_no_evidence_reply(
        message: models.Message,
        current_draft_text: str | None,
        evidence_status: str,
) -> GeneratedReply:
    """Acknowledge the citizen's message without stating any administrative fact.

    A fixed template is safe but useless to the agent: it says nothing about
    this specific request. Generating under a no-facts contract keeps the same
    safety property while producing something worth editing.
    """
    unanswered = [
        "Aucune source officielle pertinente n'a été trouvée : les éléments "
        "factuels (délais, conditions, montants, démarches) restent à vérifier "
        "et à rédiger par un agent."
    ]
    prompt = _build_prompt(message, current_draft_text).replace(
        "Draft reply:\n\n", f"{NO_EVIDENCE_CONTRACT}\nDraft reply:\n\n"
    )
    try:
        body = (AIService().call_ai_api(prompt, temperature=0.2) or "").strip()
    except Exception:  # noqa: BLE001 - never fail the endpoint on this path
        logger.exception("No-evidence reply generation failed; using fixed template.")
        body = ""

    return GeneratedReply(
        body=body or SAFE_DRAFT_BODY,
        needs_human_review=True,
        unanswered_points=unanswered,
        evidence_status=evidence_status,
        grounded=False,
    )


def generate_ai_reply_body_with_rag(
        message: models.Message, current_draft_text: str | None = None
) -> GeneratedReply:
    """Generate a reply grounded in reranked Albert excerpts.

    Pipeline: build the query from the thread, retrieve a broad candidate set,
    rerank it, drop weak evidence, then either generate under a grounding
    contract or acknowledge the request without stating any fact.
    """
    query = _rag_search_query(message, current_draft_text)
    evidence = ai_rag.retrieve_evidence(query)

    if not evidence.has_evidence:
        if settings.AI_RAG_REQUIRE_EVIDENCE:
            # No supporting excerpt: never state administrative facts from the
            # model's own memory. Acknowledge, and hand the facts to the agent.
            logger.info(
                "No usable evidence (%s) for message %s; acknowledgement only.",
                evidence.status.value,
                message.id,
            )
            return _generate_no_evidence_reply(
                message, current_draft_text, evidence.status.value
            )
        # Evidence not required (explicitly configured): ungrounded reply,
        # still flagged for review.
        body = AIService().call_ai_api(_build_prompt(message, current_draft_text))
        return GeneratedReply(
            body=body.strip(),
            needs_human_review=True,
            evidence_status=evidence.status.value,
            grounded=False,
        )

    context_block = build_rag_context(query, evidence.chunks)
    prompt = _build_grounded_prompt(message, current_draft_text, context_block)
    raw = AIService().call_ai_api(prompt, temperature=0.2)

    parsed = _parse_structured_reply(raw)
    if parsed is None:
        # The model ignored the output contract. The text may still be a fine
        # reply, but we cannot verify its grounding, so it needs review.
        logger.warning(
            "AI reply for message %s did not follow the JSON contract.", message.id
        )
        reply = GeneratedReply(body=(raw or "").strip(), needs_human_review=True)
    else:
        reply = _validate_structured_reply(parsed, len(evidence.chunks))

    if not reply.body:
        raise ValueError("AI response does not contain a reply body")

    reply.evidence_status = evidence.status.value
    reply.sources = evidence.sources_metadata()
    reply.grounded = True
    return reply


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
            reply = GeneratedReply(
                body=generate_preview_reply_body(source_message),
                evidence_status="preview",
                needs_human_review=True,
            )
        else:
            try:
                reply = generate_ai_reply_body_with_rag(
                    source_message, current_draft_text
                )
            except ImproperlyConfigured:
                logger.info(
                    "AI service is not configured; creating preview AI draft for message %s",
                    message_id,
                )
                reply = GeneratedReply(
                    body=generate_preview_reply_body(source_message),
                    evidence_status=ai_rag.EvidenceStatus.NOT_CONFIGURED.value,
                    needs_human_review=True,
                )
            except Exception as exc:
                logger.exception("Failed to generate AI draft for message %s", message_id)
                raise ServiceUnavailable("Failed to generate AI draft.") from exc

        ai_metadata = reply.metadata()
        # Logged with the draft id so a bad answer can be investigated later
        # (which excerpts, which model, which prompt version).
        logger.info("AI draft generated for message %s: %s", message_id, ai_metadata)

        draft = create_draft(
            mailbox=sender_mailbox,
            subject=_reply_subject(source_message.subject),
            draft_body=_blocknote_paragraphs(reply.body),
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
        payload = dict(
            serializers.MessageSerializer(draft, context={"request": request}).data
        )
        # Additive key: no schema migration, no breaking change for clients
        # that ignore it. The composer can surface it when the UI is ready.
        payload["aiMetadata"] = ai_metadata
        return Response(payload, status=status.HTTP_201_CREATED)

"""Inspect the retrieval pipeline stage by stage for one citizen email.

This exists because the pipeline can fail silently in four different ways that
all look identical from the mailbox (a holding reply): search error, zero
results, reranker error, or every chunk falling under the relevance threshold.

Usage::

    python manage.py ai_rag_debug --text "Bonjour, je me suis inscrit..."
    python manage.py ai_rag_debug --file email.txt
    python manage.py ai_rag_debug --file email.txt --no-rerank
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.test import override_settings

from core.services.ai_rag import retrieve_evidence
from core.services.ai_service import AIService


class Command(BaseCommand):
    """Print what search returned, what rerank scored, and what the gate kept."""

    help = "Debug the Albert retrieval pipeline for a single email text."

    def add_arguments(self, parser):
        """Declare CLI arguments."""
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--text", help="Raw email text to use as the query.")
        group.add_argument("--file", help="Path to a file containing the email text.")
        parser.add_argument(
            "--no-rerank", action="store_true", help="Skip the reranking stage."
        )
        parser.add_argument(
            "--excerpts",
            action="store_true",
            help="Print the excerpt text, not only the scores.",
        )

    def handle(self, *args, **options):
        """Run each stage and report it."""
        if options["file"]:
            path = Path(options["file"])
            if not path.is_file():
                raise CommandError(f"File not found: {path}")
            query = path.read_text(encoding="utf-8")
        else:
            query = options["text"]
        query = (query or "").strip()
        if not query:
            raise CommandError("Empty query.")

        self._print_config()

        overrides = {"AI_RAG_RERANK_MODEL": ""} if options["no_rerank"] else {}
        with override_settings(**overrides):
            self._stage_search(query, options["excerpts"])
            self._stage_rerank(query)
            self._stage_gate(query, options["excerpts"])

    def _print_config(self):
        """Show the settings that actually drive the pipeline."""
        self.stdout.write(self.style.MIGRATE_HEADING("Configuration"))
        for name in (
            "AI_BASE_URL",
            "AI_MODEL",
            "AI_SEARCH_METHOD",
            "AI_COLLECTION_IDS",
            "AI_PRIVATE_COLLECTION_ID",
            "AI_RAG_SEARCH_CANDIDATES",
            "AI_RAG_CONTEXT_LIMIT",
            "AI_RAG_RERANK_MODEL",
            "AI_RAG_MIN_RELEVANCE_SCORE",
            "AI_RAG_REQUIRE_EVIDENCE",
        ):
            value = getattr(settings, name, None)
            if name == "AI_COLLECTION_IDS" and value:
                value = f"{value} (types: {[type(v).__name__ for v in value]})"
            self.stdout.write(f"  {name:<28}= {value}")
        self.stdout.write("")

    def _stage_search(self, query, show_excerpts):
        """Stage 1: raw vector search."""
        self.stdout.write(self.style.MIGRATE_HEADING("Stage 1 - POST /v1/search"))
        try:
            chunks = AIService().search_chunks(
                query, limit=settings.AI_RAG_SEARCH_CANDIDATES
            )
        except Exception as exc:  # noqa: BLE001 - this command reports failures
            self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))
            self.stdout.write(
                self.style.WARNING(
                    "  -> This is why you get the holding reply. Check the "
                    "collection ids, the API key, and the search limit.\n"
                )
            )
            self._candidates = []
            return

        self._candidates = chunks
        if not chunks:
            self.stdout.write(self.style.WARNING("  0 chunk returned."))
            self.stdout.write(
                self.style.WARNING(
                    "  -> This is why you get the holding reply. The query or "
                    "the collections are the problem, not the gate.\n"
                )
            )
            return

        self.stdout.write(f"  {len(chunks)} candidates")
        for index, chunk in enumerate(chunks[:10], start=1):
            self.stdout.write(
                f"    {index:>2}. score={chunk.get('search_score')} "
                f"doc={chunk.get('document_name') or chunk.get('document_id')}"
            )
            if show_excerpts:
                self.stdout.write(f"        {chunk['content'][:200]}...")
        self.stdout.write("")

    def _stage_rerank(self, query):
        """Stage 2: reranking, with the raw score distribution."""
        self.stdout.write(self.style.MIGRATE_HEADING("Stage 2 - POST /v1/rerank"))
        if not settings.AI_RAG_RERANK_MODEL:
            self.stdout.write("  Disabled (AI_RAG_RERANK_MODEL is empty).\n")
            return
        if not self._candidates:
            self.stdout.write("  Skipped (no candidates).\n")
            return

        documents = [chunk["content"] for chunk in self._candidates]
        try:
            results = AIService().rerank(
                query, documents, settings.AI_RAG_CONTEXT_LIMIT
            )
        except Exception as exc:  # noqa: BLE001 - this command reports failures
            self.stdout.write(self.style.ERROR(f"  FAILED: {exc}"))
            self.stdout.write(
                "  -> Not fatal: the pipeline falls back to search order.\n"
            )
            return

        if not results:
            self.stdout.write(self.style.WARNING("  0 result returned.\n"))
            return

        scores = [r.get("relevance_score") for r in results]
        for result in results:
            self.stdout.write(
                f"    index={result['index']:>2} "
                f"relevance_score={result.get('relevance_score')}"
            )
        numeric = [s for s in scores if isinstance(s, (int, float))]
        if numeric:
            self.stdout.write(
                f"\n  Score range: {min(numeric):.4f} .. {max(numeric):.4f}"
            )
            threshold = settings.AI_RAG_MIN_RELEVANCE_SCORE
            if threshold > 0 and max(numeric) < threshold:
                self.stdout.write(
                    self.style.ERROR(
                        f"  -> EVERY score is below AI_RAG_MIN_RELEVANCE_SCORE "
                        f"({threshold}). This is why you get the holding reply. "
                        f"Lower it to 0 or just under {min(numeric):.4f}."
                    )
                )
            else:
                self.stdout.write(
                    f"  Use this range to pick AI_RAG_MIN_RELEVANCE_SCORE "
                    f"(currently {threshold})."
                )
        self.stdout.write("")

    def _stage_gate(self, query, show_excerpts):
        """Stage 3: the decision the draft endpoint will actually make."""
        self.stdout.write(self.style.MIGRATE_HEADING("Stage 3 - evidence gate"))
        evidence = retrieve_evidence(query)
        self.stdout.write(f"  status   : {evidence.status.value}")
        self.stdout.write(f"  reranked : {evidence.reranked}")
        self.stdout.write(f"  kept     : {len(evidence.chunks)} excerpts")

        if not evidence.has_evidence:
            self.stdout.write(
                self.style.ERROR(
                    "\n  -> The draft endpoint WILL return the holding reply.\n"
                )
            )
            return

        for source in evidence.sources_metadata():
            self.stdout.write(
                f"    [{source['index']}] rerank={source['rerank_score']} "
                f"search={source['search_score']} "
                f"doc={source['document_name'] or source['document_id']}"
            )
            if show_excerpts:
                self.stdout.write(f"        {source['excerpt'][:300]}...")
        self.stdout.write(
            self.style.SUCCESS("\n  -> The draft endpoint WILL generate a reply.\n")
        )

"""Evaluate the AI reply retrieval pipeline against a small corpus.

Albert's own guidance is that reranking may be counterproductive on a given
corpus, so this command exists to measure it rather than assume it.

Corpus format (JSON list)::

    [
      {
        "id": "carte-identite-delai",
        "question": "Bonjour, quel est le délai pour refaire ma carte d'identité ?",
        "expected_keywords": ["délai", "instruction"],
        "expected_document_ids": ["doc-123"],
        "expect_human_review": false
      }
    ]

Usage::

    python manage.py ai_draft_eval --corpus core/evaluation/ai_draft_corpus.json
    python manage.py ai_draft_eval --corpus ... --no-rerank   # baseline
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.test import override_settings

from core.services.ai_rag import EvidenceStatus, retrieve_evidence


class Command(BaseCommand):
    """Run the retrieval pipeline over a corpus and report retrieval quality."""

    help = "Evaluate the AI draft retrieval pipeline on a corpus of citizen emails."

    def add_arguments(self, parser):
        """Declare CLI arguments."""
        parser.add_argument("--corpus", required=True, help="Path to the JSON corpus.")
        parser.add_argument(
            "--no-rerank",
            action="store_true",
            help="Disable reranking, to compare against the raw search baseline.",
        )
        parser.add_argument(
            "--json", action="store_true", help="Emit machine-readable results."
        )

    def handle(self, *args, **options):
        """Run the evaluation."""
        path = Path(options["corpus"])
        if not path.is_file():
            raise CommandError(f"Corpus not found: {path}")

        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CommandError(f"Invalid JSON corpus: {exc}") from exc
        if not isinstance(cases, list) or not cases:
            raise CommandError("Corpus must be a non-empty JSON list.")

        overrides = {"AI_RAG_RERANK_MODEL": ""} if options["no_rerank"] else {}
        with override_settings(**overrides):
            results = [self._evaluate(case) for case in cases]

        if options["json"]:
            self.stdout.write(json.dumps(results, ensure_ascii=False, indent=2))
            return

        self._report(results, reranked=not options["no_rerank"])

    def _evaluate(self, case: dict) -> dict:
        """Run one case through the retrieval pipeline."""
        question = (case.get("question") or "").strip()
        evidence = retrieve_evidence(question)

        excerpts = " ".join(chunk["content"].lower() for chunk in evidence.chunks)
        keywords = [kw.lower() for kw in case.get("expected_keywords") or []]
        hits = [kw for kw in keywords if kw in excerpts]

        found_docs = {
            str(chunk.get("document_id")) for chunk in evidence.chunks
        }
        expected_docs = {str(d) for d in case.get("expected_document_ids") or []}

        expects_review = bool(case.get("expect_human_review"))
        got_review = not evidence.has_evidence

        return {
            "id": case.get("id") or question[:40],
            "status": evidence.status.value,
            "candidates": evidence.candidates_count,
            "kept": len(evidence.chunks),
            "reranked": evidence.reranked,
            "keyword_recall": (len(hits) / len(keywords)) if keywords else None,
            "missing_keywords": [kw for kw in keywords if kw not in hits],
            "document_recall": (
                len(expected_docs & found_docs) / len(expected_docs)
                if expected_docs
                else None
            ),
            "review_expected": expects_review,
            "review_triggered": got_review,
            "review_correct": expects_review == got_review,
            "top_rerank_score": (
                evidence.chunks[0].get("rerank_score") if evidence.chunks else None
            ),
        }

    def _report(self, results: list[dict], reranked: bool) -> None:
        """Print a human-readable summary."""
        total = len(results)
        answered = sum(1 for r in results if r["status"] == EvidenceStatus.OK.value)
        review_correct = sum(1 for r in results if r["review_correct"])
        keyword_scores = [
            r["keyword_recall"] for r in results if r["keyword_recall"] is not None
        ]
        doc_scores = [
            r["document_recall"] for r in results if r["document_recall"] is not None
        ]

        mode = "reranked" if reranked else "search-only baseline"
        self.stdout.write(self.style.MIGRATE_HEADING(f"\nPipeline: {mode}"))
        self.stdout.write(f"Cases                : {total}")
        self.stdout.write(f"Answered with evidence: {answered}/{total}")
        self.stdout.write(f"Correct review gate  : {review_correct}/{total}")
        if keyword_scores:
            self.stdout.write(
                f"Mean keyword recall  : {sum(keyword_scores) / len(keyword_scores):.2f}"
            )
        if doc_scores:
            self.stdout.write(
                f"Mean document recall : {sum(doc_scores) / len(doc_scores):.2f}"
            )

        failures = [r for r in results if not r["review_correct"]]
        if failures:
            self.stdout.write(self.style.WARNING("\nReview-gate mismatches:"))
            for result in failures:
                self.stdout.write(
                    f"  - {result['id']}: status={result['status']} "
                    f"expected_review={result['review_expected']}"
                )

        weak = [
            r
            for r in results
            if r["keyword_recall"] is not None and r["keyword_recall"] < 1
        ]
        if weak:
            self.stdout.write(self.style.WARNING("\nIncomplete keyword coverage:"))
            for result in weak:
                self.stdout.write(
                    f"  - {result['id']}: missing {result['missing_keywords']}"
                )
        self.stdout.write("")

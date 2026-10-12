"""What docs/adr/0003-exact-deduplication.md claims.

The deduplication, and the copies in the top 10 before it (`run_ids.PRE_DEDUP_RUN`).
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import corpus, corpus_changes, datasets, retrieval_earlier
from ...run_ids import PRE_DEDUP_RUN
from ...sources import Sources
from ..comparisons import duplicate_share_text, top_k_text

# The question whose round-1 pool the record uses as its example.
EXAMPLE_QUESTION = "cv-0001"


def exact_deduplication_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0003-exact-deduplication.md."""
    _groups(doc, src)
    _copies_in_top_k(doc, src)
    _decision(doc, src)
    _consequences(doc, src)


def _groups(doc: Doc, src: Sources) -> None:
    """Context: the groups of byte-identical text."""
    dup = corpus.duplicates(src)
    doc.check(
        "Of the {} chunks the corpus produced, {} ({}%) belonged to {} groups of "
        "byte-identical text; {} of the groups were Binance",
        formatting.count(dup.occurrences),
        formatting.count(dup.in_groups),
        duplicate_share_text(src),
        formatting.count(dup.groups),
        formatting.count(dup.by_venue["binance"]),
    )
    doc.claim(
        "no group spanned {} venues",
        formatting.count(corpus.CROSS_VENUE_VENUES),
        holds=corpus.no_cross_venue_texts(src),
        what=f"no byte-identical text on {formatting.count(corpus.CROSS_VENUE_VENUES)} venues",
    )


def _copies_in_top_k(doc: Doc, src: Sources) -> None:
    """Context: copies of one text in each method's top 10, and in a labeling pool."""
    k = top_k_text(src)
    doc.ref("each method's top 10 (`results/{}`)", PRE_DEDUP_RUN)
    doc.check("each method's top {} (`results/", k)
    doc.check(
        "Measured over the {} selected questions",
        formatting.count(retrieval_earlier.copy_slot_questions(src)),
    )
    doc.check(
        "held a mean of {}--{} slots per question in each method's top {}.",
        *[formatting.metric(v) for v in retrieval_earlier.copy_slot_mean_range(src)],
        k,
    )
    doc.check(
        "The means were bm25 {}, dense {}, hybrid {} and hybrid_rerank {}.",
        *[formatting.metric(v) for v in retrieval_earlier.copy_slot_means(src)],
    )
    doc.check(
        "This happened on {}--{} questions per method, with up to {} slots on one question.",
        *[formatting.count(v) for v in retrieval_earlier.copy_question_range(src)],
        formatting.count(retrieval_earlier.copy_slot_most(src)),
    )
    doc.check(
        "was judged on {} candidates holding {} distinct texts",
        *[formatting.count(v) for v in datasets.migrated_pool(src, EXAMPLE_QUESTION)],
    )


def _decision(doc: Doc, src: Sources) -> None:
    doc.check(
        "Identical text on {} venues is never merged",
        formatting.count(corpus.CROSS_VENUE_VENUES),
    )
    doc.check("all {} old ids mapped", formatting.count(corpus_changes.dedup_old_ids(src)))


def _consequences(doc: Doc, src: Sources) -> None:
    doc.check(
        "The index holds {} chunks ({}) and {} tokens ({}).",
        formatting.count(corpus.chunk_count(src)),
        formatting.count(corpus_changes.dedup_chunk_change(src)),
        formatting.count(corpus.token_total(src)),
        formatting.count(corpus_changes.dedup_token_change(src)),
    )
    doc.definition(
        "a trap made of two identical chunks",
        reason="the two questions' trap as written, one chunk per environment",
    )

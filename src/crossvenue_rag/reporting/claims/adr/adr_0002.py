"""What docs/adr/0002-structural-chunking.md claims.

The corpus's size, the chunking config, and the measurements made before exact deduplication
with an earlier chunker.
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import corpus, corpus_changes
from ...sources import Sources
from ..reasons import BGE_WINDOW

BEFORE_DEDUP = (
    "measured on the corpus before exact deduplication with the earlier chunker; "
    "neither is committed"
)


def structural_chunking_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0002-structural-chunking.md."""
    _context(doc, src)
    _decision(doc, src)


def _context(doc: Doc, src: Sources) -> None:
    doc.definition("cuts signatures in half", reason="a description of fixed-window splitting")
    doc.check(
        "the {} chunks of `data/processed/chunks.jsonl` hold {} tokens (`token_count`, "
        "counted with that tokenizer) and {} whitespace-separated words, {} tokens per word",
        formatting.count(corpus.chunk_count(src)),
        formatting.count(corpus.token_total(src)),
        formatting.count(corpus.word_total(src)),
        formatting.ratio(corpus.tokens_per_word(src), formatting.TOKENS_PER_WORD_DECIMALS),
    )
    doc.given(
        'A "{}-word" chunk therefore overruns a {}-token encoder window',
        formatting.count(corpus.chunk_token_budget(src)),
        formatting.count(BGE_WINDOW),
        reason="chunking.max_tokens and the encoder's window",
    )
    doc.definition(
        "had three defects", reason="the list of defects that follows in the same paragraph"
    )
    doc.given("With them, {} prose chunks", "124", reason=BEFORE_DEDUP)
    doc.check(
        "of the {} the corpus had before deduplication",
        formatting.count(corpus_changes.dedup_old_ids(src)),
    )
    doc.given(
        "reduced that to {}.",
        "18",
        reason="measured on the corpus before exact deduplication; that corpus is not committed",
    )


def _decision(doc: Doc, src: Sources) -> None:
    doc.given(
        "`min_tokens` ({}) of body",
        formatting.count(src.default_config.chunking.min_tokens),
        reason="chunking.min_tokens",
    )
    doc.given(
        "`max_tokens` ({}) up to `max_code_block_tokens` ({})",
        formatting.count(corpus.chunk_token_budget(src)),
        formatting.count(src.default_config.chunking.max_code_block_tokens),
        reason="chunking.max_tokens and chunking.max_code_block_tokens",
    )
    doc.given(
        "Overlap ({} tokens)",
        formatting.count(src.default_config.chunking.overlap_tokens),
        reason="chunking.overlap_tokens",
    )

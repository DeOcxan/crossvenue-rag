"""Tests for `crossvenue_rag.corpus.chunking`."""

from __future__ import annotations

from crossvenue_rag.config_pipeline import ChunkingConfig
from crossvenue_rag.corpus.chunking import Chunker
from crossvenue_rag.models import Document, make_chunk_id
from tests.conftest import count_words


def make_doc(text: str, venue: str = "bybit") -> Document:
    return Document(
        venue=venue,
        venue_type="cex",
        doc_path="doc.md",
        source_url="https://example.invalid/doc.md",
        text=text,
    )


def small_cfg(**patch) -> ChunkingConfig:
    base = {
        "max_tokens": 40,
        "overlap_tokens": 8,
        "min_tokens": 10,
        "keep_code_blocks_intact": True,
        "max_code_block_tokens": 100,
    }
    return ChunkingConfig(**{**base, **patch})


LONG_SECTION = "# Place Order\n\n## Body\n\n" + "\n\n".join(
    f"Paragraph {i} about order placement and parameters and limits." for i in range(12)
)


CODE_DOC = (
    "# Example\n\nCall the endpoint:\n\n```python\n"
    + "\n".join(f"line_{i} = compute(value_{i})" for i in range(25))
    + "\n```\n\nThen inspect the response.\n"
)


def test_heading_breadcrumb_is_carried_into_continuation_chunks():
    """Chunk 3 of a long section must still say which endpoint it describes."""
    chunks = Chunker(small_cfg(), count_words).chunk(make_doc(LONG_SECTION))
    assert len(chunks) > 1
    later = chunks[-1]
    assert later.heading_path == ["Place Order", "Body"]
    assert later.text.startswith("Place Order > Body")


def test_short_heading_sections_are_merged_into_their_body():
    """`## Parameters` alone must not become a chunk."""
    text = "# Endpoint\n\n## Parameters\n\nsymbol is required and must be upper case.\n"
    chunks = Chunker(small_cfg(), count_words).chunk(make_doc(text))
    assert len(chunks) == 1
    assert "Parameters" in chunks[0].text
    assert "symbol is required" in chunks[0].text


def test_a_long_section_is_split_with_overlap_and_stays_within_budget():
    cfg = small_cfg(overlap_tokens=12)
    chunks = Chunker(cfg, count_words).chunk(make_doc(LONG_SECTION))
    assert len(chunks) >= 3
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    # Some paragraph from the tail of chunk 1 reappears at the head of chunk 2.
    assert set(chunks[0].text.split("\n\n")) & set(chunks[1].text.split("\n\n"))


def test_a_code_block_is_kept_whole_and_out_of_the_overlap():
    """An oversized chunk is a smaller sin than a signature cut in half, and a
    code sample duplicated into the next chunk is pure index bloat."""
    cfg = small_cfg(max_code_block_tokens=1000, overlap_tokens=30)
    chunks = Chunker(cfg, count_words).chunk(make_doc(CODE_DOC))
    with_code = [c for c in chunks if "line_0 = compute(value_0)" in c.text]
    assert len(with_code) == 1
    assert "line_24" in with_code[0].text


def test_an_oversized_code_block_is_split_on_line_boundaries():
    cfg = small_cfg(max_code_block_tokens=20)
    chunks = Chunker(cfg, count_words).chunk(make_doc(CODE_DOC))
    pieces = [c for c in chunks if "line_" in c.text]
    assert len(pieces) > 1
    # Splitting happened between lines, so no statement is cut in half.
    for chunk in pieces:
        for line in chunk.text.split("\n"):
            if line.startswith("line_"):
                assert line.endswith(")")


def test_a_single_overlong_line_is_split_on_words():
    """Real sources contain them: one PancakeSwap farm list is, after normalization, a
    5,661-character table row. Left whole it becomes a chunk whose text past the
    encoder's 512 tokens is never embedded."""
    cfg = small_cfg()
    one_line = "# Farms\n\n" + " ".join(f"token_{i}" for i in range(200)) + "\n"
    chunks = Chunker(cfg, count_words).chunk(make_doc(one_line))
    assert len(chunks) > 1
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    assert "token_199" in chunks[-1].text


def test_chunk_ids_are_stable_across_runs():
    """Gold labels have to stay valid through re-ingestion of an unchanged corpus."""
    doc = make_doc(LONG_SECTION)
    first = Chunker(small_cfg(), count_words).chunk(doc)
    second = Chunker(small_cfg(), count_words).chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    # Pinned: an id that depended on the process (a salted hash) would differ here.
    assert [c.chunk_id for c in first][:2] == ["a82ddb82de6dfa87", "f593fc4080d546a9"]


def test_chunk_ids_are_content_hashes_that_never_cross_a_venue():
    """Identical text on two venues must get distinct ids: which venue says it
    is the whole measurement. Identical text within one venue gets the *same* id
    wherever it is: that is what deduplication collapses on."""
    text = "# Rate Limits\n\nThe limit is 1200 requests per minute.\n"
    a = Chunker(small_cfg(), count_words).chunk(make_doc(text, venue="binance"))
    b = Chunker(small_cfg(), count_words).chunk(make_doc(text, venue="bybit"))
    assert a[0].text == b[0].text
    assert a[0].chunk_id != b[0].chunk_id
    assert make_chunk_id("bybit", "same") == make_chunk_id("bybit", "same")
    assert make_chunk_id("bybit", "same") != make_chunk_id("binance", "same")


def test_each_chunk_records_its_single_occurrence_as_a_source():
    text = "# Rate Limits\n\nThe limit is 1200 requests per minute.\n"
    [chunk] = Chunker(small_cfg(), count_words).chunk(make_doc(text, venue="binance"))
    [source] = chunk.sources
    assert (source.venue, source.doc_path, source.position) == (
        chunk.venue,
        chunk.doc_path,
        chunk.position,
    )
    assert source.heading_path == chunk.heading_path

"""Structure-aware chunking of API documentation under a token budget.

A fixed-window splitter cuts a parameter table away from its endpoint and a
signature in half. This chunker breaks at headings once a chunk reaches
`min_tokens` and keeps a fenced code block whole up to `max_code_block_tokens`. A
chunk whose body does not start with a heading gets its heading breadcrumb as a
prefix; one that starts with a heading keeps that heading alone (docs/reference/corpus.md).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ..config_pipeline import ChunkingConfig
from ..models import Chunk, ChunkSource, Document, make_chunk_id
from .normalize import split_code_blocks

TokenCounter = Callable[[str], int]

# Markdown's deepest heading level.
_MAX_HEADING_DEPTH = 6
# An overlong word's width shrinks to three quarters per try: few tries, little overshoot.
_SHRINK_NUMERATOR, _SHRINK_DENOMINATOR = 3, 4
_HEADING_RE = re.compile(rf"^(?P<hashes>#{{1,{_MAX_HEADING_DEPTH}}})\s+(?P<title>.*?)\s*#*$")


@dataclass(slots=True)
class Block:
    """A markdown block with the heading path in effect where it starts."""

    kind: str  # "heading" | "code" | "text"
    text: str
    heading_path: list[str]


def parse_blocks(markdown: str) -> list[Block]:
    """The blocks of normalized markdown, each with its heading breadcrumb.

    Blank lines separate prose blocks; fenced code is one block regardless of the
    blank lines inside it.
    """
    blocks: list[Block] = []
    path: list[str] = []

    for segment in split_code_blocks(markdown):
        if segment.is_code:
            if segment.text.strip():
                blocks.append(Block("code", segment.text.strip("\n"), list(path)))
            continue

        buffer: list[str] = []

        def flush(buffer: list[str] = buffer, path: list[str] = path) -> None:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(Block("text", body, list(path)))
            buffer.clear()

        for line in segment.text.split("\n"):
            heading = _HEADING_RE.match(line)
            if heading:
                flush()
                level = len(heading.group("hashes"))
                title = heading.group("title").strip()
                del path[level - 1 :]
                while len(path) < level - 1:
                    path.append("")  # a document that jumps h1 -> h3
                path.append(title)
                blocks.append(Block("heading", line.strip(), list(path)))
            elif not line.strip():
                flush()
            else:
                buffer.append(line)
        flush()

    return blocks


def _split_oversized(
    block: Block, max_tokens: int, count: TokenCounter, cfg: ChunkingConfig
) -> Iterator[Block]:
    """Breaks a single block that cannot fit in a chunk on its own.

    Code blocks are kept whole up to ``max_code_block_tokens``: an oversized
    chunk is a smaller sin than an unusable one. Above that they are split on
    line boundaries, which keeps statements intact.

    Prose is split on lines first and, only when a single line does not fit, on
    words. Some sources have such lines: a PancakeSwap farm list is, after
    normalization, one 5,661-character table row of 3,736 tokens. Left whole it
    becomes a chunk of which a 512-token encoder silently sees about the first seventh.
    """
    tokens = count(block.text)
    if tokens <= max_tokens:
        yield block
        return

    if block.kind == "code" and cfg.keep_code_blocks_intact and tokens <= cfg.max_code_block_tokens:
        yield block
        return

    def emit(pieces: list[str], separator: str) -> Block:
        return Block(block.kind, separator.join(pieces), list(block.heading_path))

    current: list[str] = []
    for line in block.text.split("\n"):
        if count(line) > max_tokens:
            if current:
                yield emit(current, "\n")
                current = []
            yield from _split_on_words(line, max_tokens, count, emit)
            continue
        candidate = [*current, line]
        if current and count("\n".join(candidate)) > max_tokens:
            yield emit(current, "\n")
            current = [line]
        else:
            current = candidate
    if current:
        yield emit(current, "\n")


def _split_on_words(
    line: str,
    max_tokens: int,
    count: TokenCounter,
    emit: Callable[[list[str], str], Block],
) -> Iterator[Block]:
    """Last-resort split of one unreasonably long line, on whitespace.

    A "word" that does not fit is split on characters. Only space-free junk
    reaches that case (a table rule hundreds of columns wide), but without it a
    single token blob can break the budget outright.
    """
    current: list[str] = []
    for word in line.split(" "):
        for piece in _fit_on_characters(word, max_tokens, count):
            candidate = [*current, piece]
            if current and count(" ".join(candidate)) > max_tokens:
                yield emit(current, " ")
                current = [piece]
            else:
                current = candidate
    if current:
        yield emit(current, " ")


def _fit_on_characters(word: str, max_tokens: int, count: TokenCounter) -> list[str]:
    if count(word) <= max_tokens:
        return [word]
    # The first width comes from the observed characters-per-token ratio, so the
    # shrinking takes two or three iterations in practice.
    width = max(1, int(len(word) * max_tokens / max(1, count(word))))
    while width > 1 and count(word[:width]) > max_tokens:
        width = width * _SHRINK_NUMERATOR // _SHRINK_DENOMINATOR
    return [word[i : i + width] for i in range(0, len(word), width)]


def _breadcrumb(path: list[str]) -> str:
    parts = [p for p in path if p]
    return " > ".join(parts)


class Chunker:
    """Packs blocks into chunks under a token budget.

    `count_tokens` is the embedding model's own tokenizer in production, so a
    450-token chunk is 450 tokens to the model that encodes it. Tests pass a
    whitespace word count, which keeps the model out of the test suite.
    """

    def __init__(self, cfg: ChunkingConfig, count_tokens: TokenCounter) -> None:
        """A chunker with the budgets in `cfg`, counting tokens with `count_tokens`."""
        self.cfg = cfg
        self.count = count_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        """The chunks of `doc`, in document order."""
        blocks: list[Block] = []
        for block in parse_blocks(doc.text):
            blocks.extend(_split_oversized(block, self._split_budget(block), self.count, self.cfg))

        chunks: list[Chunk] = []
        current: list[Block] = []
        pending: bool = False  # a block was added since the last flush

        def flush(*, seed_overlap: bool) -> None:
            nonlocal current, pending
            # Two things that are not chunks: a heading with no body under it
            # ("## Parameters" alone), and a chunk made only of the overlap
            # carried over from the previous one, which would be pure duplicate.
            if not pending or all(b.kind == "heading" for b in current):
                return
            chunk = self._materialise(doc, current, len(chunks))
            if chunk is not None:
                chunks.append(chunk)
            current = self._overlap_tail(current) if seed_overlap else []
            pending = False

        for block in blocks:
            # A heading starts another chunk, but only once the running chunk is
            # substantial; this is what merges "## Parameters" into its body.
            if (
                block.kind == "heading"
                and current
                and self._body_tokens(current) >= self.cfg.min_tokens
            ):
                flush(seed_overlap=False)

            if current and self._tokens_of([*current, block]) > self.cfg.max_tokens:
                flush(seed_overlap=True)
                # The flush may have left carried-over overlap behind, or been a
                # no-op because `current` was *only* carried-over overlap. Either
                # way, if the block still does not fit, the overlap is dropped:
                # losing context is better than silently exceeding the budget,
                # and the heading stays so the chunk still says where it is.
                if current and self._tokens_of([*current, block]) > self.cfg.max_tokens:
                    current = [b for b in current if b.kind == "heading"]

            current.append(block)
            pending = True

        flush(seed_overlap=False)
        return chunks

    def _tokens_of(self, blocks: list[Block]) -> int:
        """Token count of the rendered chunk these blocks become.

        The heading breadcrumb `_render` prepends counts, and tokenizers are not
        additive, so a sum of separately counted blocks undershoots the joined
        text by the separators. Either error would push chunks past `max_tokens`.
        """
        return self.count(self._render(blocks))

    def _body_tokens(self, blocks: list[Block]) -> int:
        """Token count of the blocks alone, without the heading breadcrumb.

        `min_tokens` and `overlap_tokens` are about how much *document* a chunk
        carries; counting the breadcrumb toward them would let a deep heading
        path fill an overlap budget entirely and silently switch overlap off.
        """
        return self.count("\n\n".join(b.text for b in blocks))

    def _split_budget(self, block: Block) -> int:
        """Budget for splitting one oversized block, before its chunk is known.

        Conservative: the piece may end up heading-led or breadcrumbed, so the budget
        holds back room for the longer of the two prefixes.
        """
        crumb = _breadcrumb(block.heading_path)
        if not crumb:
            return self.cfg.max_tokens
        # `+ len(path)` covers the `#` markers when the prefix is a real heading
        # line rather than the breadcrumb rendering of the same words.
        reserve = self.count(crumb + "\n\n") + len([p for p in block.heading_path if p])
        return max(1, self.cfg.max_tokens - reserve)

    def _render(self, blocks: list[Block]) -> str:
        """The exact text these blocks become, used for both the budget and the output."""
        if not blocks:
            return ""
        body = "\n\n".join(b.text for b in blocks).strip()
        if not body:
            return ""
        crumb = _breadcrumb(blocks[0].heading_path)
        # The breadcrumb is part of the indexed text: without it, chunk 3 of a
        # long "Place Order" section does not say what it is about.
        return f"{crumb}\n\n{body}" if crumb and not body.startswith("#") else body

    def _overlap_tail(self, blocks: list[Block]) -> list[Block]:
        """Trailing blocks of the flushed chunk that fit in the overlap budget.

        Overlap is taken in whole blocks rather than sliced tokens: a trailing
        paragraph restores context for the next chunk, and half a sentence does
        not.
        """
        if self.cfg.overlap_tokens <= 0:
            return []
        tail: list[Block] = []
        for block in reversed(blocks):
            if block.kind == "code":
                break  # a code sample is never duplicated into the next chunk
            candidate = [block, *tail]
            if self._body_tokens(candidate) > self.cfg.overlap_tokens:
                break
            tail = candidate
        return tail

    def _materialise(self, doc: Document, blocks: list[Block], position: int) -> Chunk | None:
        text = self._render(blocks)
        if not text:
            return None
        heading_path = [p for p in blocks[0].heading_path if p]

        return Chunk(
            chunk_id=make_chunk_id(doc.venue, text),
            venue=doc.venue,
            venue_type=doc.venue_type,
            doc_path=doc.doc_path,
            heading_path=heading_path,
            source_url=doc.source_url,
            text=text,
            token_count=self.count(text),
            position=position,
            # One occurrence per chunk here; `dedup` merges occurrences of the
            # same text into one chunk after the whole corpus is chunked.
            sources=[
                ChunkSource(
                    venue=doc.venue,
                    doc_path=doc.doc_path,
                    heading_path=list(heading_path),
                    source_url=doc.source_url,
                    position=position,
                )
            ],
        )

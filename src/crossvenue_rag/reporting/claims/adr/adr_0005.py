"""What docs/adr/0005-labeling-pool.md claims.

The pool's depth and size, the corpus size and the run's k.
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import corpus, datasets_rounds, retrieval
from ...run_ids import POOL_RUN
from ...sources import Sources
from ..comparisons import top_k_text
from ..methodology import protocol

DEPTH = "eval.pool_depth_per_method"
DEPTH_AND_DISPLAY = "eval.pool_depth_per_method and eval.pool_display"


def labeling_pool_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0005-labeling-pool.md."""
    _context(doc, src)
    _decision(doc, src)
    _consequences(doc, src)


def _context(doc: Doc, src: Sources) -> None:
    """The title's pool parameters, then the context."""
    depth, display = protocol.pool_depth_text(src), protocol.pool_display_text(src)
    doc.given("methods, depth {}, {} shown", depth, display, reason=DEPTH_AND_DISPLAY)
    doc.check("reading {} chunks per question", formatting.count(corpus.chunk_count(src)))
    doc.check("so part of its top {} on comparison questions", top_k_text(src))


def _decision(doc: Doc, src: Sources) -> None:
    depth, display = protocol.pool_depth_text(src), protocol.pool_display_text(src)
    doc.given(
        "method's top {} (`eval.pool_depth_per_method`), fused with RRF, and the top {} "
        "distinct candidates are shown",
        depth,
        display,
        reason=DEPTH_AND_DISPLAY,
    )
    doc.given(
        "fewer than {} remain, each method is asked for more, up to {} times the depth",
        display,
        formatting.count(datasets_rounds.pool_depth_multiplier(src)),
        reason="eval.pool_display and labeling.pool._MAX_DEPTH_MULTIPLIER",
    )
    doc.given("has its unjudged top {} on those questions", depth, reason=DEPTH)
    doc.definition(
        "Sakai 2007, [doi:10.1145/1277741.1277756]", reason="a citation: author, year and DOI"
    )


def _consequences(doc: Doc, src: Sources) -> None:
    k = top_k_text(src)
    doc.check(
        "pooled from the {} methods that existed then",
        formatting.count(len(retrieval.ROUND1_POOL)),
    )
    doc.given(
        "venue_split_oracle's never-judged top {} (`results/",
        protocol.pool_depth_text(src),
        reason=DEPTH,
    )
    doc.ref("never-judged top 20 (`results/{}`)", POOL_RUN)
    doc.check("For a method whose top {} is fully judged", k)
    doc.check("how much of each pool method's top {} is judged", k)

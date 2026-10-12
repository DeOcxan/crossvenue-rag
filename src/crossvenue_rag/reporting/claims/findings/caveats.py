"""docs/results/findings.md: the limits on the retrieval results."""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import datasets, retrieval, retrieval_tables
from ...sources import Sources
from ..comparisons import comparison_questions

# The labeled categories the caveats count, in the order the text lists them.
SMALL_CATEGORIES = ("adversarial", "fact", "ambiguous", "noisy", "version_sensitive", "multihop")


def caveats(doc: Doc, src: Sources) -> None:
    """The caveats of the retrieval results."""
    doc.check(
        "the labeled categories have {}--{} questions (adversarial {}, fact {}, ambiguous {}, "
        "noisy {}, version_sensitive {}, multihop {})",
        *[formatting.count(v) for v in retrieval_tables.labelled_size_range(src, SMALL_CATEGORIES)],
        *[formatting.count(v) for v in retrieval_tables.labelled_sizes(src, SMALL_CATEGORIES)],
    )
    doc.check(
        "{} of the {} comparison questions have a venue recorded",
        formatting.count(datasets.one_sided_comparisons(src)),
        comparison_questions(src),
    )
    doc.check(
        "venue_split_oracle's unjudged@10 is {}; judging",
        formatting.metric(retrieval.oracle_unjudged(src, "ambiguous")),
    )

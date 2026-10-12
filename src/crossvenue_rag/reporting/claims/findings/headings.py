"""docs/results/findings.md: the results headings, each stating its result with a number.

Style rule 16 has a results heading state its result; these are the headings whose
number no section's own check covers.
"""

from __future__ import annotations

from .... import formatting
from ....generation.answer_rows import mean_column
from ...check import Doc
from ...compute import (
    answers,
    answers_judged,
    datasets,
    documents,
    retrieval,
)
from ...compute import retrieval_off_venue as off_venue
from ...compute.answers import PROMPT_V1
from ...compute.answers_costs import MERCURY
from ...sources import Sources
from ..comparisons import comparison_questions, duplicate_share_text, significance, top_k_text

# The tail of the limits heading, after its count, which the check below compares.
LIMITS = " limits on the retrieval results"
# The judged summary column of coverage on the comparison questions.
COVERAGE = mean_column("coverage", "comparison")


def headings(doc: Doc, src: Sources) -> None:
    """Results headings that state their result with a number (rule 16).

    Raises:
        ValueError: `doc`'s text has no `LIMITS`, or no heading after it.
    """
    doc.check(
        "### Venue detection finds every named venue on {} of {} comparison questions",
        formatting.count(datasets.detection_counts(src, "comparison")["exact"]),
        comparison_questions(src),
    )
    doc.claim(
        "### Adding {} gold chunks changes every method's comparison recall@10 by {} to {}",
        formatting.count(datasets.augmentation(src).relevant),
        *[formatting.signed(v) for v in retrieval.recall_change_range(src)],
        holds=retrieval.recall_lower_for_every_method(src),
        what=f"every method's comparison {retrieval.at_top_k(src, 'recall')} is lower "
        "against the augmented gold",
    )
    share = retrieval.at_top_k(src, "off_venue")
    doc.check(
        "### Unnamed venues fill {} of hybrid_rerank's comparison top {} and {} of BM25's "
        "(exploratory)",
        formatting.metric(off_venue.off_venue_means(src, "hybrid_rerank", "comparison")[1][share]),
        top_k_text(src),
        formatting.metric(off_venue.off_venue_means(src, "bm25", "comparison")[1][share]),
    )
    doc.check(
        "### The cross-encoder takes {} of hybrid_rerank's {} ms at p50",
        formatting.ms(retrieval.step_p50(src, "hybrid_rerank", "rerank")),
        formatting.ms(retrieval.summary_metric(src, "hybrid_rerank", "p50_ms")),
    )
    doc.check(
        "### {} limits on the retrieval results",
        formatting.count(len(documents.list_items(doc.text, LIMITS, "\n#"))),
    )
    doc.check(
        "## {}% of chunk occurrences are byte-identical",
        duplicate_share_text(src),
    )
    doc.claim(
        "### Qwen has the highest coverage ({}) and Mercury refuses {} of {}",
        formatting.metric(
            answers_judged.judged_summary(src, PROMPT_V1, "qwen@venue_split", COVERAGE)
        ),
        formatting.count(answers.comparison_refusals(src, MERCURY)),
        comparison_questions(src),
        holds=answers_judged.ahead_of_every_pair(src, "qwen@venue_split", "coverage", "comparison"),
        what="Qwen's comparison coverage is above every other configuration's, each pair at "
        f"p_Holm < {significance()}",
    )

"""docs/results/findings.md, its references: every run, `index_version` and hash it names.

Each is the one the claims read.
"""

from __future__ import annotations

from .... import formatting
from ....config import SCHEMA_DEFAULTS
from ...check import Doc
from ...compute import answers_judged, corpus, registry
from ...run_ids import (
    ANSWERS_V1,
    ANSWERS_V2,
    BENCH_RUN,
    GOLD,
    JUDGED_V1,
    JUDGED_V2,
    NDCG_PRIME_RUN,
    ONE_CALL_RUN,
    POWER_RUN,
    PROMPTS_RUN,
    PVALUE_RUN,
    RETRIEVAL_RUN,
)
from ...sources import Sources
from ..comparisons import (
    comparison_questions,
    file_short_hash,
    level,
    question_count_text,
    top_k_text,
)

# Every run findings.md cites, in the order it is checked.
CITED_RUNS = (
    RETRIEVAL_RUN,
    NDCG_PRIME_RUN,
    PVALUE_RUN,
    ONE_CALL_RUN,
    BENCH_RUN,
    POWER_RUN,
    ANSWERS_V1,
    ANSWERS_V2,
    JUDGED_V1,
    JUDGED_V2,
    PROMPTS_RUN,
)


def runs(doc: Doc, src: Sources) -> None:
    """Every run findings.md names, and where it names it."""
    version = corpus.index_version(src)
    for run, current in registry.runs_on_index(src, CITED_RUNS):
        doc.claim(
            f"`results/{run}`",
            holds=current,
            what=f"results/{run} was made against the current index {version}",
        )
    doc.ref(
        "`index_version` `{}`, manifest `{}`, in full in `evals/datasets/v0.2.1-gold.meta.json`",
        version,
        file_short_hash(src, SCHEMA_DEFAULTS.paths.manifest),
    )
    doc.ref(
        "k = {}, `index_version` `{}`, questions file sha256 `{}`",
        top_k_text(src),
        version,
        file_short_hash(src, GOLD),
    )
    doc.ref("Run `results/{}`: ", RETRIEVAL_RUN)
    doc.ref("nDCG'@10 is from `results/{}`", NDCG_PRIME_RUN)
    doc.ref("are from `results/{}`;", PVALUE_RUN)
    doc.ref("without adjustment (`results/{}`)", POWER_RUN)
    doc.ref("every metric but latency in `results/{}`, are identical", ONE_CALL_RUN)
    doc.ref("Run `results/{}` (`crossvenue bench-rerank`)", BENCH_RUN)
    doc.ref("Prompt v1 answers are `results/{}`", ANSWERS_V1)
    doc.ref("which Opus judged in `results/{}`. Only", JUDGED_V1)
    doc.ref("run `results/{}`, `calibration.json`", ANSWERS_V1)
    doc.ref("Answers `results/{}` (the same", ANSWERS_V2)
    doc.ref("which Opus judged in `results/{}`; the", JUDGED_V2)
    doc.ref("comparison with v1 is `results/{}`.", PROMPTS_RUN)


def captions(doc: Doc, src: Sources) -> None:
    """Table captions (style rule 9) not written next to their table's check."""
    doc.definition(
        "Prompt v2 replaces rule 5 with three outcomes",
        reason="full answer, partial answer and refusal, the outcomes prompt v2 allows",
    )
    doc.check(
        "The detector applied to the text of the {} questions of `v0.2.1-gold`",
        question_count_text(src),
    )
    doc.caption(
        "Recall@10 on the {} comparison questions, the retrievals of `results/{run}` scored "
        "against both (deviation D4):",
        (RETRIEVAL_RUN,),
        comparison_questions(src),
    )
    doc.caption(
        "prompt v1, run `results/{run}` (`summary.csv`), `v0.2.1-gold` (deviation D4), the {} "
        "comparison and all "
        "{} questions. Each mean is over the judged answers that carry the metric: on all "
        "questions {} or {} answers for coverage, {} or {} for `behaviour_correct`, and the {} "
        "to {} answers with claims for venue_attribution_error:",
        (JUDGED_V1,),
        comparison_questions(src),
        question_count_text(src),
        *[
            formatting.count(n)
            for metric in ("coverage", "behaviour_correct", "venue_attribution_error")
            for n in answers_judged.metric_answer_range(src, metric, "all")
        ],
    )
    doc.caption(
        "is the primary comparison above. Run `results/{run}`, `v0.2.1-gold`, a - b "
        "over the n questions where both answers carry claims, among the {} comparison and all "
        "{} questions, {}% paired bootstrap intervals, p_Holm within each scope",
        (JUDGED_V1,),
        comparison_questions(src),
        question_count_text(src),
        level(),
    )
    doc.caption(
        "Run `results/{run}`, `v0.2.1-gold`, the {} comparison questions (a - b), {}% paired "
        "bootstrap intervals (deviations D4 and D16):",
        (JUDGED_V1,),
        comparison_questions(src),
        level(),
    )
    doc.ref(
        "Runs `results/{}` and `results/{}`, `v0.2.1-gold`, n per category (deviations D10 and "
        "D11):",
        ANSWERS_V1,
        ANSWERS_V2,
    )
    doc.ref("H-v2, v2 - v1 per configuration, run `results/{}`, `v0.2.1-gold`", PROMPTS_RUN)
    doc.ref(
        "Judged runs `results/{}` (v1) and `results/{}` (v2), `v0.2.1-gold`, answers with claims "
        "per row (deviations D10 and D14):",
        JUDGED_V1,
        JUDGED_V2,
    )

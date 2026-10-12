"""docs/results/findings.md: the retrieval results on `v0.2.1-gold`, from the header on."""

from __future__ import annotations

from .... import formatting
from ....config_eval import COMPARISON_SCOPE, LABELLED_COMPARISON_SCOPE, LABELLED_SCOPE
from ....evaluation import power as power_module
from ...check import Doc
from ...compute import bench, corpus, datasets, power, retrieval, retrieval_tables
from ...run_ids import (
    AFTER_RESTRUCTURE_RUN,
    BEFORE_RESTRUCTURE_RUN,
    NDCG_PRIME_RUN,
    PVALUE_RUN,
    RETRIEVAL_RUN,
)
from ...sources import Sources
from ..comparisons import (
    RERANKER,
    SPLIT,
    better_worse,
    comparison_questions,
    coverage_parts,
    estimate,
    level,
    p_bound,
    question_count_text,
    signed_ms_parts,
    signed_parts,
    significance,
    top_k_text,
)
from ..methodology.protocol import bootstrap_resamples, bootstrap_seed_text
from ..methodology.significance import DELTA
from ..reasons import INTERVAL_LEVEL

# How the comparison table names the oracle.
METHOD_LABELS = {"venue_split_oracle": "venue_split_oracle (oracle)"}


def header(doc: Doc, src: Sources) -> None:
    """The header: the index, the corpus and the gold set."""
    doc.check(
        "over the deduplicated index ({} chunks of {} trading venues and the CCXT library; the {} "
        "documents of the manifest, {} of which yield chunks;",
        formatting.count(corpus.chunk_count(src)),
        formatting.count(corpus.venue_count(src)),
        formatting.count(corpus.manifest_count(src)),
        formatting.count(corpus.chunked_document_count(src)),
    )
    doc.check(
        "({} questions, {} with gold chunks, {} gold chunks)",
        question_count_text(src),
        formatting.count(datasets.labelled_count(src)),
        formatting.count(datasets.gold_chunk_total(src)),
    )


def parameters(doc: Doc, src: Sources) -> None:
    """Results on v0.2.1-gold: the run's parameters."""
    doc.check(
        "{} methods, k = {}, `index_version`",
        formatting.count(retrieval.method_count(src)),
        top_k_text(src),
    )
    doc.check(
        "the run's `index.json`; the {} questions of `v0.2.1-gold`, {} with gold chunks, {} "
        "comparison)",
        question_count_text(src),
        formatting.count(datasets.labelled_count(src)),
        comparison_questions(src),
    )
    doc.given("Paired intervals are {}% percentile", level(), reason=INTERVAL_LEVEL)
    doc.check(
        "intervals over {} bootstrap resamples (seed {})",
        bootstrap_resamples(src),
        bootstrap_seed_text(src),
    )
    doc.check("(sign flip, {} permutations, same seed)", bootstrap_resamples(src))
    doc.given("significant only when `p_Holm` < {}.", significance(), reason="significance level")


def comparison_table(doc: Doc, src: Sources) -> None:
    """The comparison table, and the heading on venue coverage."""
    doc.caption(
        "Run `results/{run}` (nDCG'@10 from `results/{run}`), `v0.2.1-gold`, the {} comparison "
        "questions (deviations D4 and D13):",
        (RETRIEVAL_RUN, NDCG_PRIME_RUN),
        comparison_questions(src),
    )
    doc.table(
        f"(deviations D4 and D13): | method | {retrieval.at_top_k(src, 'recall')}",
        {
            METHOD_LABELS.get(r.method, r.method): [
                formatting.metric(r.recall),
                formatting.metric(r.ndcg),
                formatting.metric(r.ndcg_prime),
                f"{formatting.fixed(r.coverage, formatting.SHARE_DECIMALS)} "
                f"({formatting.count(r.covered)}/{formatting.count(r.questions)})",
            ]
            for r in retrieval_tables.comparison_rows(src)
        },
    )
    doc.check(
        "### Splitting retrieval by venue raises comparison coverage from {} to {}",
        # A heading may omit the count of a share (docs/style.md rule 11).
        formatting.fixed(retrieval.covered_share(src, "hybrid_rerank"), formatting.SHARE_DECIMALS),
        formatting.fixed(retrieval.covered_share(src, "venue_split"), formatting.SHARE_DECIMALS),
    )
    doc.check(
        "`venue_split` was designed after the `v0.2-gold` run measured the coverage failure on "
        "these {} questions, so this test is no independent confirmation of it. It fuses BM25 "
        "and dense candidates once per venue the question names, reranks them and interleaves "
        "the per-venue lists "
        "([design](../explanation/design.md#retrieval), [ADR 0007](../adr/0007-venue-split.md)).",
        comparison_questions(src),
    )
    doc.check("within the retrieved top {}.", top_k_text(src))
    doc.check("whose top {} hold at least one", top_k_text(src))


def venue_split(doc: Doc, src: Sources) -> None:
    """venue_split - hybrid_rerank on venue coverage, the primary comparison."""
    primary = (*SPLIT, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE)
    doc.check(
        "venue_coverage@10 (primary): {} [{}, {}]; better on {} questions",
        *signed_parts(retrieval.paired_interval(src, *primary)),
        formatting.count(retrieval.paired_counts(src, *primary).better),
    )
    doc.claim(
        "worse on none; McNemar p < {}, p_rand < {}.",
        p_bound(),
        p_bound(),
        holds=retrieval.paired_never_worse(src, *primary)
        and retrieval.paired_p_below(src, *primary, "mcnemar_p")
        and retrieval.paired_p_below(src, *primary, "randomisation_p"),
        what=f"venue_split worse on no question; both p-values below {p_bound()}",
    )
    doc.check(
        "`v0.2-gold` run had measured the coverage failure on the same {} questions",
        comparison_questions(src),
    )
    doc.check(
        "puts each detected venue's first chunk into the top {}, so coverage is", top_k_text(src)
    )
    doc.check(
        "For venue_split and the oracle, whose top {} on comparison is fully", top_k_text(src)
    )
    doc.check("which leaves part of each one's own top {}", top_k_text(src))
    by_detection, by_fallback = retrieval.coverage_by_detection(src)
    doc.check(
        "guaranteed whenever detection finds every named venue. Of the {} covered comparison "
        "questions, {} are covered that way; the other {} fall back to",
        formatting.count(retrieval.coverage_counts(src, "venue_split")[0]),
        formatting.count(by_detection),
        formatting.count(by_fallback),
    )


def split_recall(doc: Doc, src: Sources) -> None:
    """venue_split - hybrid_rerank on recall, and the power to detect a gap."""
    recall = (*SPLIT, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE)
    split = retrieval.paired_counts(src, *recall)
    doc.claim(
        "recall@10 (secondary): {} [{}, {}], p_Holm = {}; better on {}, worse on {}, tied on {}; "
        "inconclusive at this size.",
        *signed_parts(retrieval.paired_interval(src, *recall)),
        formatting.p_value(retrieval.paired_p(src, *recall, "holm_p")),
        *[formatting.count(v) for v in (split.better, split.worse, split.ties)],
        holds=retrieval.paired_includes_zero(src, *recall),
        what=f"the {retrieval.at_top_k(src, 'recall')} interval includes 0",
    )
    study = src.power
    doc.check(
        "is a paired randomization on per-question recall@10, which is fractional for {} of the "
        "family's {} values, with Holm within a family of {}. A simulation with the family's "
        "observed discordance ({} of {} question pairs differ) reaches {}% power from a gap of {} "
        "when the comparison is Holm's first step and from {} without adjustment",
        formatting.count(study.fractional_values),
        formatting.count(study.values),
        formatting.count(power.family_size(src)),
        formatting.count(study.discordant_pairs),
        formatting.count(study.question_pairs),
        formatting.count(power_module.TARGET_POWER_PERCENT),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        formatting.design_parameter(power.detectable_gap(src, "unadjusted")),
    )
    gap = retrieval.paired_gap(src, *recall)
    doc.claim(
        "The observed gap of {} is below the {} it detects.",
        formatting.metric(gap),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        holds=power.below_detectable_gap(src, gap, "holm_first_step"),
        what=f"the observed recall gap is below the "
        f"{formatting.count(power_module.TARGET_POWER_PERCENT)}%-power gap",
    )


def split_cost(doc: Doc, src: Sources) -> None:
    """venue_split's latency against hybrid_rerank's, and the rerank time of per-venue calls."""
    doc.check(
        "Over all {} questions p50 is {} ms against {} ms for hybrid_rerank, "
        "p95 {} ms against {} ms. "
        "On the {} questions venue_split splits, p50 is {} ms against {} ms for hybrid_rerank on "
        "the same questions",
        question_count_text(src),
        formatting.ms(retrieval.summary_metric(src, "venue_split", "p50_ms")),
        formatting.ms(retrieval.summary_metric(src, "hybrid_rerank", "p50_ms")),
        formatting.ms(retrieval.summary_metric(src, "venue_split", "p95_ms")),
        formatting.ms(retrieval.summary_metric(src, "hybrid_rerank", "p95_ms")),
        formatting.count(len(datasets.split_ids(src))),
        formatting.ms(retrieval.split_p50(src, "venue_split")),
        formatting.ms(retrieval.split_p50(src, "hybrid_rerank")),
    )
    doc.check(
        "On those questions per-venue calls \u2212 one call is {} ms [{}, {}] per query in the "
        "rerank step (bench, descriptive)",
        *signed_ms_parts(bench.paired_difference(src, "rerank_ms")),
    )


def reranker(doc: Doc, src: Sources) -> None:
    """The section on hybrid_rerank against hybrid: the losses excluded, the power, the table."""
    recall = (*RERANKER, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE)
    coverage = (*RERANKER, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE)
    doc.check(
        "### Against hybrid, a recall@10 loss above {} from the reranker is excluded at {}%",
        formatting.metric(retrieval.excluded_loss(src, *recall)),
        level(),
    )
    doc.caption(
        "Paired differences with {}% paired bootstrap intervals, run `results/{run}`, "
        "`v0.2.1-gold`, the {} comparison questions (deviations D4 and D16):",
        (PVALUE_RUN,),
        level(),
        comparison_questions(src),
    )
    doc.check(
        "Against hybrid, a loss larger than {} in recall@10 or {} in venue_coverage@10 is "
        "excluded at {}%",
        formatting.metric(retrieval.excluded_loss(src, *recall)),
        formatting.metric(retrieval.excluded_loss(src, *coverage)),
        level(),
    )
    doc.claim(
        "and both gains are inconclusive at this size (recall@10 reaches {}% power from a gap "
        "of {})",
        formatting.count(power_module.TARGET_POWER_PERCENT),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        holds=retrieval.paired_includes_zero(src, *recall)
        and retrieval.paired_includes_zero(src, *coverage),
        what="both hybrid_rerank - hybrid intervals include 0",
    )
    doc.table(
        "| hybrid_rerank - |",
        {
            r.b: [
                estimate(r.recall),
                formatting.p_value(r.recall_holm),
                estimate(r.coverage),
                formatting.p_value(r.coverage_mcnemar),
                formatting.p_value(r.coverage_holm),
            ]
            for r in retrieval_tables.reranker_rows(src)
        },
    )


def first_stage(doc: Doc, src: Sources) -> None:
    """The reranker against dense and BM25."""
    dense = ("hybrid_rerank", "dense", retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE)
    doc.check(
        "comparison gold, {} questions better and {} worse (p_Holm = {})",
        *better_worse(retrieval.paired_counts(src, *dense)),
        formatting.p_value(retrieval.paired_p(src, *dense, "holm_p")),
    )
    bm25 = ("hybrid_rerank", "bm25", retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE)
    doc.claim(
        "Against BM25 the comparison-recall difference is {} [{}, {}] ({} better, {} worse), "
        "p_Holm = {}, inconclusive at this size. The test reaches {}% power from a gap of {}.",
        *signed_parts(retrieval.paired_interval(src, *bm25)),
        *better_worse(retrieval.paired_counts(src, *bm25)),
        formatting.p_value(retrieval.paired_p(src, *bm25, "holm_p")),
        formatting.count(power_module.TARGET_POWER_PERCENT),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        holds=retrieval.paired_gap(src, *bm25) < DELTA
        and not retrieval.paired_significant(src, *bm25),
        what="hybrid_rerank - bm25 comparison recall is below "
        f"{formatting.design_parameter(DELTA)} and has p_Holm >= "
        f"{significance()}",
    )
    labelled = ("hybrid_rerank", "bm25", retrieval.at_top_k(src, "recall"), LABELLED_SCOPE)
    doc.check(
        "Over all {} labeled questions the gain is {} [{}, {}] (p_Holm = {})",
        formatting.count(retrieval.paired_questions(src, *labelled)),
        *signed_parts(retrieval.paired_interval(src, *labelled)),
        formatting.p_value(retrieval.paired_p(src, *labelled, "holm_p")),
    )
    coverage = (
        "hybrid_rerank",
        "dense",
        retrieval.at_top_k(src, "venue_coverage"),
        COMPARISON_SCOPE,
    )
    doc.claim(
        "Dense retrieval's venue_coverage@10 is {} ({}/{}); hybrid_rerank - "
        "dense on venue_coverage@10 is {} [{}, {}] ({} vs {} discordant questions), p_Holm = {}, "
        "inconclusive at this size.",
        *coverage_parts(src, "dense"),
        *signed_parts(retrieval.paired_interval(src, *coverage)),
        *better_worse(retrieval.paired_counts(src, *coverage)),
        formatting.p_value(retrieval.paired_p(src, *coverage, "holm_p")),
        holds=not retrieval.paired_significant(src, *coverage),
        what=f"hybrid_rerank - dense on comparison coverage has p_Holm >= {significance()}",
    )


def restructuring(doc: Doc, src: Sources) -> None:
    """The runs made before and after the move into subpackages, compared on every metric."""
    doc.ref("Run `results/{}`, made after the code moved", AFTER_RESTRUCTURE_RUN)
    doc.ref("and run `results/{}`, made before it", BEFORE_RESTRUCTURE_RUN)
    doc.claim(
        "both on the {} questions of `v0.2.1-gold`, have the same value in every metric column",
        question_count_text(src),
        holds=retrieval.runs_agree_on_metrics(src, AFTER_RESTRUCTURE_RUN, BEFORE_RESTRUCTURE_RUN),
        what="the runs before and after the restructuring agree on every metric column",
    )

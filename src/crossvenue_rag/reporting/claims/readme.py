"""What README.md claims.

Its functions, each for a section or a results row, run in check order.
"""

from __future__ import annotations

from typing import Protocol

from ... import formatting
from ...config_eval import COMPARISON_SCOPE, LABELLED_COMPARISON_SCOPE
from ...dataset.agreement import KAPPA_TARGET
from ...evaluation import power as power_module
from ...evaluation.comparisons import TOP_ONE_OFF_VENUE
from ..check import Doc
from ..compute import (
    agreement,
    answers,
    bench,
    calibration,
    corpus,
    corpus_pancakeswap,
    corpus_sheet,
    datasets,
    power,
    registry,
    repository,
    retrieval,
    sensitivity,
)
from ..compute import retrieval_off_venue as off_venue
from ..compute.answers import PROMPT_V1, PROMPT_V2
from ..outcomes import Param
from ..run_ids import (
    ANSWERS_V1,
    JUDGED_V1,
    JUDGED_V2,
    POWER_RUN,
    RETRIEVAL_RUN,
    SENSITIVITY_BASE_RUN,
    SENSITIVITY_RUN,
)
from ..sources import Sources
from .comparisons import (
    RERANKER,
    SPLIT,
    comparison_questions,
    coverage_parts,
    duplicate_share_text,
    kappa_bounds,
    level,
    p_bound,
    question_count_text,
    sensitivity_sizes,
    signed_parts,
    top_k_text,
)
from .reasons import INTERVAL_LEVEL

# The wall time of `make test`, which no saved run holds.
TEST_WALL_TIME_S = "70"
TEST_WALL_TIME = (
    "the median wall time of `make test`, rounded up: three runs of 59.34 to 73.31 s, median "
    "65.69 s, on 2026-10-10; no saved run holds it"
)
# The wall time of `make check`: lint, every test with the slow one, and reproduce.
CHECK_WALL_TIME_S = "270"
CHECK_WALL_TIME = (
    "the wall time of `make check`, rounded up: one run of 261.77 s on 2026-10-10; "
    "no saved run holds it"
)
JUDGE_MODEL = "anthropic/claude-opus-5.5"
# The count of values `make reproduce` compares across every checked document, table cells
# included, which README.md states. No document's claims can recompute it, so `reproduce`
# compares it with its own total once every document has been checked.
CHECKED_NUMBERS = "2,582"
CHECKED_NUMBERS_TEMPLATE = (
    "compares {} values of the checked documents with their sources, table cells included"
)
CHECKED_NUMBERS_REASON = "the values `make reproduce` compares, checked against its own total"


class TestCounter(Protocol):
    """Counts the tests pytest collects: those `make test` runs, or with `every` all of them."""

    def __call__(self, *, every: bool) -> int:
        """The count of the tests `make test` runs, or with `every` of all tests."""
        ...


def readme(doc: Doc, src: Sources, test_count: TestCounter) -> None:
    """States the claims of README.md, with `test_count` counting the tests it states.

    `test_count(every=False)` counts the tests `make test` runs, `every=True` all of them.
    The count of values `make reproduce` compares is checked by `reproduce` itself, once
    every document has been checked (`CHECKED_NUMBERS`).
    """
    _pitch(doc, src)
    _gold_set(doc, src)
    _coverage_row(doc, src)
    _reranker_row(doc, src)
    _pooling_row(doc, src)
    _latency_row(doc, src)
    _round_two(doc, src)
    _deduplication_row(doc, src)
    _pancakeswap_row(doc, src)
    _answers_rows(doc, src)
    _requirements(doc, src)
    _tests(doc, src, test_count)
    _off_venue_row(doc, src)
    _references(doc, src)
    _scope(doc, src)
    _checked_numbers(doc)
    _quick_start(doc, src)
    _documentation(doc, src)


def _pitch(doc: Doc, src: Sources) -> None:
    """The opening paragraph's headline result and the figure's caption."""
    doc.check(
        "retrieving once per named venue (`venue_split`) raises the share of the {} comparison "
        "questions with every named venue in the top {} from {} to {}. The difference is "
        "{} [{}, {}] (",
        comparison_questions(src),
        top_k_text(src),
        *coverage_parts(src, "hybrid_rerank")[:1],
        *coverage_parts(src, "venue_split")[:1],
        *signed_parts(
            retrieval.paired_interval(
                src, *SPLIT, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE
            )
        ),
    )
    doc.ref("In `results/{}` on `v0.2.1-gold`, retrieving once", RETRIEVAL_RUN)
    doc.given("] ({}% paired bootstrap), McNemar p", level(), reason=INTERVAL_LEVEL)
    doc.claim(
        "McNemar p < {}. `venue_split` was designed",
        p_bound(),
        holds=_split_p_below(src),
        what=f"the primary comparison's McNemar p is below {p_bound()}",
    )
    doc.ref("on the comparison questions of `v0.2.1-gold`, run `results/{}`.", RETRIEVAL_RUN)


def _split_p_below(src: Sources) -> bool:
    """Whether the primary comparison's McNemar p is below the bound README states."""
    return retrieval.paired_p_below(
        src, *SPLIT, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE, "mcnemar_p"
    )


def _gold_set(doc: Doc, src: Sources) -> None:
    doc.check(
        "`v0.2.1-gold` ({} questions, {} with gold chunks)",
        question_count_text(src),
        formatting.count(datasets.labelled_count(src)),
    )
    doc.given(
        f"`results/{JUDGED_V1}`; {{}}% paired bootstrap intervals;",
        level(),
        reason=INTERVAL_LEVEL,
    )
    doc.check(
        "n is the {} comparison questions where a row names no other.",
        comparison_questions(src),
    )


def _coverage_row(doc: Doc, src: Sources) -> None:
    comparison = comparison_questions(src)
    doc.claim(
        "| {} ({}/{}) with `venue_split` against {} ({}/{}) with hybrid_rerank: the share of "
        "comparison questions with every named venue in the top {}, {} [{}, {}], McNemar p < {}. "
        "venue_split retrieves once per named venue, and coverage is guaranteed when detection "
        "finds every named venue ({} of {}).",
        *coverage_parts(src, "venue_split"),
        *coverage_parts(src, "hybrid_rerank"),
        top_k_text(src),
        *signed_parts(
            retrieval.paired_interval(
                src, *SPLIT, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE
            )
        ),
        p_bound(),
        formatting.count(retrieval.coverage_by_detection(src)[0]),
        comparison,
        holds=_split_p_below(src),
        what=f"the primary comparison's McNemar p is below {p_bound()}",
    )
    doc.claim(
        "Recall@10 {} [{}, {}]: inconclusive at this size, with {}% power from a gap of {}, per",
        *signed_parts(
            retrieval.paired_interval(
                src, *SPLIT, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE
            )
        ),
        formatting.count(power_module.TARGET_POWER_PERCENT),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        holds=retrieval.paired_includes_zero(
            src, *SPLIT, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE
        ),
        what=f"the {retrieval.at_top_k(src, 'recall')} interval includes 0",
    )
    doc.ref(", per `results/{}`. |", POWER_RUN)


def _reranker_row(doc: Doc, src: Sources) -> None:
    doc.check(
        "| {} [{}, {}] recall@10 for hybrid_rerank - hybrid on the {} comparison questions: a "
        "recall@10 loss above {} from the reranker is excluded at {}%. venue_coverage@10 {} "
        "[{}, {}]: a loss above {} is excluded at {}%.",
        *signed_parts(
            retrieval.paired_interval(
                src, *RERANKER, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE
            )
        ),
        comparison_questions(src),
        formatting.metric(
            retrieval.excluded_loss(
                src, *RERANKER, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE
            )
        ),
        level(),
        *signed_parts(
            retrieval.paired_interval(
                src, *RERANKER, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE
            )
        ),
        formatting.metric(
            retrieval.excluded_loss(
                src, *RERANKER, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE
            )
        ),
        level(),
    )


def _pooling_row(doc: Doc, src: Sources) -> None:
    augmented = datasets.augmentation(src)
    depth = formatting.count(retrieval.pool_depth(src))
    doc.check(
        "| {} to {}: the change in every method's comparison recall@10 from `v0.2-gold` to "
        "`v0.2.1-gold`, where judging venue_split's and the oracle's unjudged top {} adds {} "
        "gold chunks, {} from ranks {}--{}.",
        *[formatting.signed(v) for v in retrieval.recall_change_range(src)],
        depth,
        formatting.count(augmented.relevant),
        formatting.count(augmented.relevant_lower),
        formatting.count(retrieval.first_rank_below_top_k(src)),
        depth,
    )


def _latency_row(doc: Doc, src: Sources) -> None:
    doc.check(
        "| {} ms of a {} ms p50 query for hybrid_rerank is the cross-encoder, over the {} "
        "questions; venue_split p50 {} ms.",
        formatting.ms(retrieval.step_p50(src, "hybrid_rerank", "rerank")),
        formatting.ms(retrieval.summary_metric(src, "hybrid_rerank", "p50_ms")),
        question_count_text(src),
        formatting.ms(retrieval.summary_metric(src, "venue_split", "p50_ms")),
    )


def _round_two(doc: Doc, src: Sources) -> None:
    """The Status line on round 2 and the Limitations bullet on agreement."""
    registered = agreement.registered(src)
    kappa = formatting.metric(registered.report.kappa)
    pairs, questions = (
        formatting.count(n) for n in (registered.report.pairs, registered.report.questions)
    )
    doc.check(
        "reproduced the round-1 labels at \u03ba {} [{}, {}] ({}% percentile bootstrap over "
        "questions) over {} decisions on {} questions",
        kappa,
        *kappa_bounds(registered),
        Param(level(), INTERVAL_LEVEL),
        pairs,
        questions,
    )
    doc.ref(
        "the adjudicated `v0.2.2-gold` (`results/{}`, against `results/{}` on",
        SENSITIVITY_RUN,
        SENSITIVITY_BASE_RUN,
    )
    doc.claim(
        "changes no retrieval comparison's sign, significance or whether its interval excludes {}",
        "0",
        holds=sensitivity.conclusions_unchanged(src),
        what="every primary and secondary comparison keeps sign, significance and whether its "
        "interval excludes 0",
    )
    doc.check(
        "The comparisons are on the {} comparison questions, the {} multi-venue ones and the "
        "labeled ones, {} in `v0.2.2-gold` and {} in `v0.2.1-gold`.",
        *sensitivity_sizes(src),
    )
    doc.claim(
        "Round 2 reproduced the labels at \u03ba {} over {} decisions on {} questions, below the "
        "registered target of {},",
        kappa,
        pairs,
        questions,
        formatting.own_form(KAPPA_TARGET),
        holds=registered.report.kappa < KAPPA_TARGET,
        what="the registered kappa is below the target",
    )


def _deduplication_row(doc: Doc, src: Sources) -> None:
    dup = corpus.duplicates(src)
    doc.check(
        "| {}% of chunk occurrences are byte-identical, {} of the {} groups Binance mainnet "
        "against",
        duplicate_share_text(src),
        formatting.count(corpus_sheet.testnet_mirror_groups(src)),
        formatting.count(dup.groups),
    )
    doc.check(
        "{} occurrences stored as {} chunks",
        formatting.count(dup.occurrences),
        formatting.count(corpus.chunk_count(src)),
    )


def _pancakeswap_row(doc: Doc, src: Sources) -> None:
    doc.check(
        "| {} median best-match TF-IDF cosine to Uniswap over the {} PancakeSwap chunks",
        formatting.metric(corpus_pancakeswap.pancakeswap_median_cosine(src)),
        formatting.count(len(corpus.venue_chunks(src, "pancakeswap"))),
    )
    bound = formatting.own_form(
        corpus_pancakeswap.PANCAKESWAP_COSINE_BOUND, formatting.COSINE_THRESHOLD_DECIMALS
    )
    doc.claim(
        "and no chunk above {}:",
        bound,
        holds=corpus_pancakeswap.pancakeswap_below_bound(src),
        what=f"no PancakeSwap chunk has best-match cosine >= {bound}",
    )


def _answers_rows(doc: Doc, src: Sources) -> None:
    judge = src.judged_v1.config.generation.chosen_judge()
    pair = answers.primary_pair(src)
    mean_a, mean_b = answers.judged_means(src, pair, "coverage", "comparison")
    doc.check(
        "| {} over venue_split against {} over hybrid_rerank: gemini's coverage of the comparison "
        "questions under prompt v1, {} [{}, {}], p_Holm = {}.",
        formatting.metric(mean_a),
        formatting.metric(mean_b),
        *signed_parts(answers.judged_interval(src, PROMPT_V1, pair, "coverage", "comparison")),
        formatting.p_value(
            answers.judged_p(src, PROMPT_V1, pair, "coverage", "comparison", "holm_p")
        ),
    )
    doc.claim(
        "_judged`: {} [{}, {}], p_Holm = {}, inconclusive at this size",
        *signed_parts(answers.judged_interval(src, PROMPT_V2, pair, "coverage", "comparison")),
        formatting.p_value(answers.primary_holm(src, PROMPT_V2, pair, "coverage", "comparison")),
        holds=answers.judged_includes_zero(src, PROMPT_V2, pair, "coverage", "comparison"),
        what="the prompt v2 coverage interval includes 0",
    )
    doc.ref("Under prompt v2 (exploratory), in `results/{}`:", JUDGED_V2)
    doc.check(
        "| {} of the {} claims the human rejects are rejected by Opus as judge too, \u03ba {}, on "
        "the {} claims of the calibration answers in",
        formatting.count(calibration.verdict_pairs(src, judge, "supported")[(False, False)]),
        formatting.count(calibration.human_negatives(src).unsupported),
        formatting.metric(
            calibration.comparison_kappa(src, calibration.judge_with_human(src), "supported")
        ),
        formatting.count(calibration.sample_claims(src)),
    )
    doc.ref("claims of the calibration answers in `results/{}`. |", ANSWERS_V1)


def _requirements(doc: Doc, src: Sources) -> None:
    doc.check("Python {} or later", repository.python_requirement(src))


def _tests(doc: Doc, src: Sources, test_count: TestCounter) -> None:
    doc.check("`make test` runs {} tests", formatting.count(test_count(every=False)))
    doc.check("all {} tests", formatting.count(test_count(every=True)))
    doc.given("tests in about {} s of wall time", TEST_WALL_TIME_S, reason=TEST_WALL_TIME)
    # The machine the bench ran on, whose machine.json records its core count.
    doc.check("on an Apple M5 with {} cores", formatting.count(bench.cpu_count(src)))
    doc.given("`crossvenue reproduce` (about {} s)", CHECK_WALL_TIME_S, reason=CHECK_WALL_TIME)


def _off_venue_row(doc: Doc, src: Sources) -> None:
    share = retrieval.at_top_k(src, "off_venue")
    doc.claim(
        "| {} of {} comparison questions put an unnamed exchange's chunk first with "
        "hybrid_rerank, {} of {} with BM25, CCXT pages counted apart (exploratory). "
        "hybrid_rerank - bm25 (both secondary): off-venue share@{} {} [{}, {}], p_Holm < {}; "
        "top-1 off-venue rate {} [{}, {}], p_Holm = {}.",
        *[
            formatting.count(v)
            for v in off_venue.first_rank_counts(
                src, TOP_ONE_OFF_VENUE, "comparison", ("hybrid_rerank", "bm25")
            )
        ],
        top_k_text(src),
        *signed_parts(off_venue.off_venue_interval(src, "bm25", share, "comparison")),
        p_bound(),
        *signed_parts(off_venue.off_venue_interval(src, "bm25", TOP_ONE_OFF_VENUE, "comparison")),
        formatting.p_value(
            off_venue.off_venue_holm_p(src, "bm25", TOP_ONE_OFF_VENUE, "comparison")
        ),
        holds=off_venue.off_venue_share_below_bound(src, "bm25", "comparison"),
        what=f"the off-venue share comparison has p_Holm below {p_bound()}",
    )


def _references(doc: Doc, src: Sources) -> None:
    doc.ref("Retrieval: `results/{}` on", RETRIEVAL_RUN)
    doc.ref("answers: `results/{}`;", JUDGED_V1)
    doc.claim(
        "answers: `results/",
        holds=registry.runs_on_current_index(src, (RETRIEVAL_RUN, JUDGED_V1)),
        what="both cited runs were made against the current index",
    )


def _scope(doc: Doc, src: Sources) -> None:
    """The opening sentence, Status and Limitations."""
    judge = src.judged_v1.config.generation.chosen_judge()
    doc.check(
        "API documentation of {} crypto trading venues (",
        formatting.count(corpus.venue_count(src)),
    )
    doc.check("indexes, {} retrieval methods", formatting.count(retrieval.method_count(src)))
    doc.check("judges the answers of {} generators", formatting.count(len(answers.generators(src))))
    doc.check(
        "The set holds {} questions, {} of them comparison questions",
        question_count_text(src),
        comparison_questions(src),
    )
    doc.claim(
        "ratings on {} of {} dimensions on the point estimate and on the interval for coverage "
        "only",
        formatting.count(calibration.passes(src, judge)),
        formatting.count(calibration.dimension_count(src)),
        holds=calibration.passes_on_interval(src, judge) == ("coverage",),
        what="coverage is the only dimension Opus passes on an informative interval",
    )
    doc.claim(
        f"Opus (`{JUDGE_MODEL}` on OpenRouter)",
        holds=answers.judge_is(src, "opus", JUDGE_MODEL),
        what="the judged run's judge is Opus with this model id",
    )


def _checked_numbers(doc: Doc) -> None:
    doc.given(CHECKED_NUMBERS_TEMPLATE, CHECKED_NUMBERS, reason=CHECKED_NUMBERS_REASON)


def _quick_start(doc: Doc, src: Sources) -> None:
    doc.check(
        "moved. The {} GitBook sources", formatting.count(corpus.sources_of_kind(src, "gitbook"))
    )


def _documentation(doc: Doc, src: Sources) -> None:
    doc.check(
        "explanation | {} development directions", formatting.count(repository.direction_count(src))
    )

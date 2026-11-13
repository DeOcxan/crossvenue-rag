"""What evals/datasheet.md claims."""

from __future__ import annotations

from typing import get_args

from ... import formatting
from ...config import SCHEMA_DEFAULTS
from ...dataset.agreement import KAPPA_TARGET
from ...dataset.journal import FIRST_ROUND
from ...dataset.validate import LEAKAGE_NGRAM
from ...evaluation.paired import MIN_TESTED_QUESTIONS
from ...evaluation.power import TARGET_POWER_PERCENT
from ...models import MULTI_VENUE, Difficulty, ExpectedBehavior, Question, fields_without_default
from ..check import Doc
from ..compute import (
    agreement,
    answers,
    corpus,
    datasets,
    datasets_rounds,
    gold_sheet,
    repository,
    round3,
)
from ..compute.estimates import DELTA
from ..outcomes import Param
from ..run_ids import CANDIDATES_FINAL, CANDIDATES_FIRST, GOLD, POOL_RUN, RETRIEVAL_RUN
from ..sources import Sources
from . import datasheet_limits, datasheet_versions
from . import round3 as round3_claims
from .comparisons import (
    comparison_questions,
    file_short_hash,
    kappa_bounds,
    level,
    optional_count,
    question_count_text,
    round2_category_text,
    round2_sample_text,
    top_k_text,
)
from .methodology import protocol
from .methodology.texts import SAMPLED, SINGLE_SAMPLED
from .reasons import CITED, INTERVAL_LEVEL, POWER_MODEL

# The venues in the order the datasheet lists how many questions name each.
VENUES = ("binance", "bybit", "hyperliquid", "bingx", "uniswap", "pancakeswap", "dydx", "ccxt")


def _category_cells(row: gold_sheet.CategoryRow) -> list[str]:
    return [
        formatting.count(n) for n in (row.questions, row.answerable, row.labelled, row.gold_chunks)
    ]


def datasheet(doc: Doc, src: Sources) -> None:
    """States the claims of evals/datasheet.md."""
    _header(doc, src)
    _motivation(doc, src)
    _composition(doc, src)
    _record_format(doc)
    _collection(doc, src)
    _labeling(doc, src)
    datasheet_versions.adjudicated_version(doc, src)
    round3_claims.version_section(doc, src)
    datasheet_limits.limitations(doc, src)
    _uses(doc, src)


def _header(doc: Doc, src: Sources) -> None:
    doc.definition("(Gebru et al., 2021, [doi:10.1145/3458723]", reason=CITED)
    doc.ref("(sha256 `{}`, in full in", file_short_hash(src, GOLD))
    doc.ref("in full in `results/{}/index.json`", RETRIEVAL_RUN)
    doc.claim(
        "byte-identical to `evals/questions.jsonl`",
        holds=repository.same_bytes(src, GOLD, SCHEMA_DEFAULTS.eval.questions),
        what="the gold set and evals/questions.jsonl have one sha256",
    )
    doc.ref(
        "| Corpus | manifest `{}` (in full", file_short_hash(src, SCHEMA_DEFAULTS.paths.manifest)
    )
    doc.check("; {} documents), `index_version`", formatting.count(corpus.manifest_count(src)))
    doc.ref("documents), `index_version` `{}`,", corpus.index_version(src))
    doc.check("`, {} chunks |", formatting.count(corpus.chunk_count(src)))
    doc.check(
        "the same {} questions without the augmentation's {} gold chunks and {} venue values",
        question_count_text(src),
        formatting.count(datasets.augmentation(src).relevant),
        formatting.count(gold_sheet.coverage_changes_to(src, "covered")),
    )


def _motivation(doc: Doc, src: Sources) -> None:
    doc.check(
        "The headline measurement is the {}-question comparison subgroup",
        comparison_questions(src),
    )
    doc.given(
        "sized to detect a recall@10 gap of {} with {}% power",
        formatting.design_parameter(DELTA),
        formatting.count(TARGET_POWER_PERCENT),
        reason=POWER_MODEL,
    )


def _composition(doc: Doc, src: Sources) -> None:
    total = datasets.question_count(src)
    answerable = gold_sheet.answerable_count(src)
    doc.check(
        "{} questions; {} answerable, {} unanswerable; {} carry gold chunks, {} gold chunks in "
        "all.",
        *[
            formatting.count(n)
            for n in (
                total,
                answerable,
                total - answerable,
                datasets.labelled_count(src),
                datasets.gold_chunk_total(src),
            )
        ],
    )
    doc.table(
        "| category | questions |",
        {c: _category_cells(gold_sheet.category_row(src, c)) for c in gold_sheet.categories(src)},
    )
    behavior = gold_sheet.field_counts(src, "expected_behavior")
    doc.check(
        "(answer {}, refuse {}, ask_clarification {})",
        *[formatting.count(behavior[b]) for b in get_args(ExpectedBehavior)],
    )
    doc.check(
        "Questions per venue named: " + ", ".join(f"{v} {{}}" for v in VENUES) + ".",
        *[formatting.count(datasets.questions_naming(src, v)) for v in VENUES],
    )
    named = gold_sheet.named_venue_counts(src)
    doc.check(
        "{} questions name one venue, {} {}, {} {} or more; {} name at least one DEX.",
        formatting.count(named.one),
        formatting.count(named.two),
        formatting.count(MULTI_VENUE),
        formatting.count(named.many),
        formatting.count(gold_sheet.MANY_VENUES),
        formatting.count(datasets.questions_naming_dex(src)),
    )
    difficulty = gold_sheet.field_counts(src, "difficulty")
    doc.check(
        "Median length {} words. Difficulty: easy {}, medium {}, hard {}.",
        formatting.ratio(datasets.median_question_words(src), formatting.COUNT_MEAN_DECIMALS),
        *[formatting.count(difficulty[d]) for d in get_args(Difficulty)],
    )
    _word_overlap(doc, src)
    doc.check(
        "{} answerable questions have no gold chunk",
        formatting.count(gold_sheet.answerable_without_gold(src)),
    )


def _word_overlap(doc: Doc, src: Sources) -> None:
    runs = gold_sheet.shared_run_counts(src)
    first, *middle, last = [[formatting.count(k), formatting.count(n)] for k, n in runs.items()]
    doc.check(
        "That run is at most {} word for {} of the {} questions with gold, "
        + ", ".join(["{} for {}"] * len(middle))
        + " and {} for {}.",
        *first,
        formatting.count(datasets.labelled_count(src)),
        *[v for pair in middle for v in pair],
        *last,
    )
    doc.claim(
        "No question reaches the {}-word threshold.",
        formatting.count(LEAKAGE_NGRAM),
        holds=gold_sheet.longest_gold_run(src) < LEAKAGE_NGRAM,
        what="every shared run is shorter than the leakage threshold",
    )
    doc.check(
        "The {} {}-word runs are", formatting.count(runs[max(runs)]), formatting.count(max(runs))
    )
    doc.definition('"403 access too frequent"', reason="a span quoted from a question")
    doc.definition("(methodology \u00a76, check 2)", reason="a check number of methodology \u00a76")


def _record_format(doc: Doc) -> None:
    doc.check(
        "The {} fields the record schema requires",
        formatting.count(len(fields_without_default(Question))),
    )


def _collection(doc: Doc, src: Sources) -> None:
    tiers = gold_sheet.field_counts(src, "tier")
    doc.check(
        "`v0.2.1-gold` holds {} `silver` and {} `bronze`",
        formatting.count(tiers["silver"]),
        formatting.count(tiers["bronze"]),
    )
    doc.check("a comparison names at least {} venues", formatting.count(MULTI_VENUE))
    candidates = gold_sheet.candidate_count(src, CANDIDATES_FINAL)
    doc.check(
        "Candidates. {} distinct candidate questions across `v0.1-candidates` ({} questions) "
        "\u2026 `v0.1.6-candidates` ({} questions)",
        formatting.count(gold_sheet.candidate_id_count(src)),
        formatting.count(gold_sheet.candidate_count(src, CANDIDATES_FIRST)),
        formatting.count(candidates),
    )
    first, final = (gold_sheet.source_split(src, p) for p in (CANDIDATES_FIRST, CANDIDATES_FINAL))
    doc.claim(
        "The answerable drafts are grounded in corpus chunks recorded as `source_chunk_ids`, "
        "{} of the {} in `v0.1-candidates` and {} of the {} in `v0.1.6-candidates`; the "
        "unanswerable ones have none",
        *(formatting.count(n) for n in (first.sourced, first.sourced + first.unsourced)),
        *(formatting.count(n) for n in (final.sourced, candidates)),
        holds=first.exactly_answerable and final.exactly_answerable,
        what="the drafts with `source_chunk_ids` are exactly the answerable ones",
    )
    doc.definition("Claude Opus 5", reason="a model name")
    decisions = gold_sheet.decisions(src)
    shown = gold_sheet.shown_count(src)
    doc.check(
        "Of the {} candidates, {} were shown: {} accepted, {} rejected",
        *[
            formatting.count(n)
            for n in (candidates, shown, decisions["accept"], decisions["reject"])
        ],
    )
    doc.check(
        "{} skipped. The other {} were never shown",
        formatting.count(decisions["skip"]),
        formatting.count(candidates - shown),
    )
    doc.check(
        "With {} rejection among the {} shown",
        formatting.count(decisions["reject"]),
        formatting.count(shown),
    )


def _labeling(doc: Doc, src: Sources) -> None:
    doc.check(
        "the full documentation of each of the {} venues",
        formatting.count(corpus.venue_count(src)),
    )
    doc.check(
        "pooled the top {} of bm25, dense, hybrid and hybrid_rerank with RRF and showed the top {}",
        protocol.pool_depth_text(src),
        protocol.pool_display_text(src),
    )
    times = datasets_rounds.save_times(src)
    median = round3.pace(src, FIRST_ROUND).per_question
    display = datasets.pool_display(src)
    doc.check(
        "median {} s per question; that median divided by {} candidates is {} s per candidate. "
        "The longer session saved {} questions over {} s, a mean of {} s between consecutive "
        "saves and {} s per candidate.",
        formatting.count(round(median)),
        protocol.pool_display_text(src),
        formatting.number(median / display, 1),
        formatting.count(times.longer_saves),
        formatting.count(round(times.longer_span)),
        formatting.number(times.longer_mean_gap, 1),
        formatting.number(times.longer_mean_gap / display, 1),
    )
    registered = agreement.registered(src)
    doc.check(
        "Round 2 reproduced these labels at \u03ba {} [{}, {}] ({}% percentile bootstrap over "
        "questions;",
        formatting.metric(registered.report.kappa),
        *kappa_bounds(registered),
        Param(level(), INTERVAL_LEVEL),
    )
    review = datasets_rounds.review_counts(src)
    doc.check(
        "Review. {} questions re-opened on their recorded round-1 pool",
        formatting.count(review.reopened),
    )
    doc.check(
        "The labels of {} changed: {} chunks added, {} removed.",
        *(formatting.count(n) for n in (review.changed, review.added, review.removed)),
    )
    _coverage(doc, src)
    _augmentation(doc, src)
    _agreement(doc, src)


def _coverage(doc: Doc, src: Sources) -> None:
    recording = gold_sheet.coverage_questions(src)
    values = gold_sheet.coverage_values(src)
    doc.claim(
        "{} values on {} questions, the {} comparison questions ({} values) and one "
        "version_sensitive question that also records a value: covered {}, "
        "missing_from_pool {}, absent_in_docs {}.",
        *[
            formatting.count(n)
            for n in (
                gold_sheet.coverage_value_count(src),
                recording.recording,
                recording.comparisons,
                recording.comparison_values,
                values["covered"],
                values["missing_from_pool"],
                values["absent_in_docs"],
            )
        ],
        holds=gold_sheet.coverage_outside_comparison(src) == ["version_sensitive"],
        what="one question outside comparison records coverage, a version_sensitive one",
    )


def _augmentation(doc: Doc, src: Sources) -> None:
    depth = protocol.pool_depth_text(src)
    doc.check(
        "On the {} comparison questions, the chunks in their top {} that round 1 never showed",
        comparison_questions(src),
        depth,
    )
    doc.check(
        "never showed ({}, {} of them in the top {})",
        formatting.count(datasets.augmentation(src).shown),
        formatting.count(datasets.augmentation(src).shown_top10),
        formatting.count(datasets.TOP_RANKS),
    )
    doc.ref("run `results/{}`)", POOL_RUN)
    added = gold_sheet.added_gold(src)
    doc.check(
        "{} were relevant, {} of them at ranks {}--{},",
        formatting.count(datasets.augmentation(src).relevant),
        formatting.count(datasets.augmentation(src).relevant_lower),
        formatting.count(datasets.TOP_RANKS + 1),
        depth,
    )
    doc.claim(
        "appended to {} questions' gold, nothing removed",
        formatting.count(len(added)),
        holds=gold_sheet.removed_gold(src) == 0,
        what="augmentation removed no gold chunk",
    )
    changes = gold_sheet.coverage_changes(src)
    doc.claim(
        "{} venue values became `covered`: {} from `missing_from_pool`, and {} from "
        "`absent_in_docs`, bingx on",
        formatting.count(gold_sheet.coverage_changes_to(src, "covered")),
        formatting.count(changes[("missing_from_pool", "covered")]),
        formatting.count(changes[("absent_in_docs", "covered")]),
        holds=gold_sheet.coverage_change_results(src) == {"covered"},
        what="every venue value augmentation changed became covered",
    )
    rank = gold_sheet.changed_venue_rank(src, "bingx", "absent_in_docs", "venue_split")
    doc.check(
        "whose answer is at rank {} of venue_split",
        optional_count(rank),
    )


def _agreement(doc: Doc, src: Sources) -> None:
    doc.check("Round 2 re-judged {} questions blind", round2_sample_text(src))
    registered = agreement.registered(src)
    low, high = kappa_bounds(registered)
    doc.claim(
        "Cohen's \u03ba between round 1 as corrected and round 2 is {} [{}, {}] over {} candidate "
        "decisions, below the registered target of {}.",
        formatting.metric(registered.report.kappa),
        low,
        high,
        formatting.count(registered.report.pairs),
        formatting.own_form(KAPPA_TARGET),
        holds=registered.report.kappa < KAPPA_TARGET,
        what="the registered kappa is below the target",
    )
    doc.check(
        "(" + ", ".join(f"{c} {{}}" for c in SAMPLED) + ", one each of",
        *[round2_category_text(src, c) for c in SAMPLED],
    )
    doc.claim(
        "one each of ambiguous, multihop, noisy and version_sensitive;",
        holds=datasets_rounds.only_single_sampled(src, SINGLE_SAMPLED),
        what="the sample holds one question of each of those categories",
    )


def _uses(doc: Doc, src: Sources) -> None:
    doc.ref("this corpus at `index_version` `{}`", corpus.index_version(src))
    doc.check("MRR at k \u2264 {} within", top_k_text(src))
    doc.check(
        "on the {} unanswerable questions is measured for {} generator configurations",
        formatting.count(datasets.category_count(src, "unanswerable")),
        formatting.count(len(answers.configurations(src))),
    )
    doc.check(
        "a category or slice of fewer than {} questions", formatting.count(MIN_TESTED_QUESTIONS)
    )

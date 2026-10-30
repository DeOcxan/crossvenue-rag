"""evals/methodology.md: sections 4 to 6, the annotator instructions among them."""

from __future__ import annotations

from .... import formatting
from ....dataset.agreement import KAPPA_TARGET, MIN_RETEST_DAYS
from ....dataset.sampling import ROUND2_SEED
from ....dataset.selection import TARGETS
from ....dataset.validate import LEAKAGE_NGRAM
from ...check import Doc
from ...compute import agreement as retest
from ...compute import (
    corpus,
    corpus_pancakeswap,
    corpus_sheet,
    datasets,
    datasets_rounds,
    gold_sheet,
)
from ...outcomes import Param
from ...run_ids import RETRIEVAL_RUN
from ...sources import Sources
from .. import comparisons
from ..reasons import DOCS_QUOTE, INTERVAL_LEVEL, LIST_FOLLOWS, REGISTERED
from . import protocol
from .texts import (
    AGREEMENT_STUDIES,
    AMBIGUITY_STUDIES,
    CONVENTION,
    DEDUPLICATION_STUDIES,
    GAP_RULES,
    INSTRUCTION_CASES,
    INSTRUCTION_GAP,
    LITERATURE,
    SAMPLED,
    SINGLE_SAMPLED,
)


def adversarial(doc: Doc, src: Sources) -> None:
    """Section 4: adversarial near-duplicate questions."""
    doc.definition(
        "Three strands are relevant, and this corpus combines all three.",
        reason="the strands that follow",
    )
    doc.definitions(AMBIGUITY_STUDIES, reason=LITERATURE)
    doc.given(
        "At least {} questions are adversarial near-duplicate questions",
        formatting.count(TARGETS["adversarial"]),
        reason=REGISTERED,
    )
    doc.check(
        "`v0.2.1-gold` has {}.", formatting.count(datasets.category_count(src, "adversarial"))
    )
    doc.ref("(run `results/{}`, exploratory", RETRIEVAL_RUN)
    doc.definition("Three sub-shapes, from the trap catalog", reason=LIST_FOLLOWS)
    _traps(doc, src)
    pairs = corpus.testnet_copies(src, "rest-api.md")
    doc.check(
        "Of the {} chunks of mainnet `rest-api.md`, {} have a byte-identical testnet occurrence "
        'and {} an identical one once the word "testnet" is removed.',
        formatting.count(pairs.chunks),
        formatting.count(pairs.identical),
        formatting.count(pairs.normalized),
    )
    doc.definition("Uniswap v2/v3/v4 use three unrelated error vocabularies.", reason=DOCS_QUOTE)
    doc.check(
        "It reports the share of each method's top {} that unnamed venues take",
        comparisons.top_k_text(src),
    )


def _traps(doc: Doc, src: Sources) -> None:
    """Section 4: each quoted value is in its venue's documentation.

    Chunks containing the term, case-insensitive, as `crossvenue count-term` counts them.
    """
    fees = comparisons.fee_tier_texts()
    for template, venue in (
        ("Uniswap's {}% v3 fee tier", "uniswap"),
        ("PancakeSwap's {}%", "pancakeswap"),
    ):
        doc.claim(
            template,
            fees[venue],
            holds=corpus_pancakeswap.fee_tier_stated(src, venue),
            what=f"{venue}'s documentation states its fee tier",
        )
    _documented(doc, src, ("Binance's `recvWindow` max of {}", "60000"), ("60000",), "binance")
    _documented(doc, src, ("BingX's max of {}", "5000"), ("5000",), "bingx")
    _documented(doc, src, ("Binance's {} ban", "418"), ("418",), "binance")
    _documented(
        doc,
        src,
        ("scales from {} minutes to {} days", "2", "3"),
        ("2 minutes", "3 days"),
        "binance",
    )
    _documented(
        doc,
        src,
        ("Bybit's {} lasts at least {} minutes", "403", "10"),
        ("at least 10 minutes", "403"),
        "bybit",
    )


def _documented(
    doc: Doc, src: Sources, quote: tuple[str, ...], terms: tuple[str, ...], venue: str
) -> None:
    """`quote` (a template and its values) holds values `venue`'s documentation contains.

    `terms` are the quoted values as the documentation writes them, one per value.
    """
    template, *values = quote
    missing = corpus.absent_terms(src, terms, venue)
    doc.claim(
        template,
        *values,
        holds=not missing,
        what=f"{venue}'s documentation contains every one of: {', '.join(terms)}",
    )


def agreement(doc: Doc, src: Sources) -> None:
    """Section 5: intra-rater reliability, and the sample it is measured on."""
    doc.definitions(AGREEMENT_STUDIES, reason=LITERATURE)
    doc.check(
        "one person labels {} questions over several sessions",
        comparisons.question_count_text(src),
    )
    doc.definition(
        "Two caveats from the reliability literature apply", reason="the two caveats that follow"
    )
    relevant, shown = datasets_rounds.first_pass_marks(src)
    doc.check(
        "(on `v0.2-gold`, {} of the {} candidates shown were marked relevant)",
        formatting.count(relevant),
        formatting.count(shown),
    )
    sample = comparisons.round2_sample_text(src)
    doc.check("- Subsample: {} of the questions round 1 judged", sample)
    doc.check(
        "questions within a category are drawn with seed {} (`dataset.sampling`)",
        formatting.own_form(ROUND2_SEED),
    )
    doc.claim(
        "On `v0.2.1-gold` that is comparison {}, unanswerable {}, adversarial {}, fact {}, and "
        "one each of ambiguous, multihop, noisy and version_sensitive.",
        *[comparisons.round2_category_text(src, c) for c in SAMPLED],
        holds=datasets_rounds.only_single_sampled(src, SINGLE_SAMPLED),
        what="the smallest categories get one sampled question each",
    )
    doc.given(
        "- Gap: \u2265 {} days between the first and second pass",
        formatting.count(MIN_RETEST_DAYS),
        reason=REGISTERED,
    )
    doc.check(
        "The order was shuffled per question with seed {}, recorded as `shuffle_seed`",
        formatting.own_form(datasets_rounds.retest_shuffle_seed(src)),
    )
    _sample_decisions(doc, src)
    _agreement_marks(doc, src)


def _sample_decisions(doc: Doc, src: Sources) -> None:
    """Section 5: the decisions of the sample, its corrections and its exposure."""
    decisions = datasets_rounds.sample_decisions(src)
    doc.check(
        "On the fixed sample that is {} decisions ({} to {} per question; {} questions carry "
        "augmentation chunks)",
        formatting.count(decisions.total),
        formatting.count(decisions.fewest),
        formatting.count(decisions.most),
        formatting.count(datasets_rounds.sample_augmented(src)),
    )
    sample = comparisons.round2_sample_text(src)
    corrected = datasets_rounds.sample_corrected(src)
    doc.check(
        "{} of the {} sampled questions carry a correction ({} from review, {} derived), and on "
        "{} of them",
        formatting.count(corrected.corrected),
        sample,
        formatting.count(corrected.review),
        formatting.count(corrected.derived),
        formatting.count(datasets_rounds.sample_correction_changes(src)),
    )
    doc.check(
        "showed in this way the gold chunks of {} of the {} sampled questions",
        formatting.count(datasets_rounds.exposed_sample_count(src)),
        sample,
    )
    doc.check(
        "The registered estimate stays \u03ba over all {} questions; \u03ba over the {} "
        "questions whose chunks were not shown",
        sample,
        formatting.count(datasets_rounds.unexposed_sample_count(src)),
    )
    doc.definition(
        "`crossvenue kappa` prints all four readings (two references, two question sets)",
        reason="the two references by the two question sets",
    )


def _agreement_marks(doc: Doc, src: Sources) -> None:
    """Section 5: how source chunks were marked, the target and the schedule."""
    marks = datasets_rounds.source_chunk_marks(src)
    doc.check(
        "Shown source chunks were marked relevant {} times out of {}, other pool chunks {} "
        "times out of {}",
        formatting.count(marks.source_relevant),
        formatting.count(marks.source_shown),
        formatting.count(marks.other_relevant),
        formatting.count(marks.other_shown),
    )
    doc.given(
        'Target \u03ba \u2265 {} (Landis & Koch "substantial")',
        formatting.own_form(KAPPA_TARGET),
        reason=REGISTERED,
    )
    doc.check(
        "| Round 1: all {} questions judged, reviewed, frozen as `v0.2-gold` |",
        formatting.count(datasets_rounds.question_count_before_augmentation(src)),
    )
    doc.check(
        "| Round-1 augmentation of the {} comparison questions |",
        comparisons.comparison_questions(src),
    )
    doc.check(
        "| Round 2 re-annotation of the {}-question subsample |",
        comparisons.round2_sample_text(src),
    )
    days = Param(formatting.count(MIN_RETEST_DAYS), REGISTERED)
    doc.check("Round 2 met the {}-day minimum under either reading", days)
    gaps = retest.study(src).gaps.values()
    span = (formatting.number(min(gaps), 1), formatting.number(max(gaps), 1))
    doc.check(
        "each question's round-1 and round-2 judgments are {} to {} days apart, from its last "
        "round-1 record",
        *span,
    )
    doc.check("round 2 ran at least fourteen days after round 1 ({} to {} days,", *span)
    registered = retest.registered(src)
    kappa = formatting.metric(registered.report.kappa)
    low, high = comparisons.kappa_bounds(registered)
    doc.check(
        "| \u03ba computed and written to `evals/agreement.md`: {} [{}, {}] ({}% percentile "
        "bootstrap over questions) |",
        kappa,
        low,
        high,
        Param(comparisons.level(), INTERVAL_LEVEL),
    )
    doc.check(
        "by either date; \u03ba is {} [{}, {}] ({}% percentile bootstrap over questions; "
        "[agreement]",
        kappa,
        low,
        high,
        Param(comparisons.level(), INTERVAL_LEVEL),
    )
    doc.check("The {}-day gap is what makes the number mean anything.", days)
    doc.check(
        "Fallback: a {}-day gap, for when {} days cannot be kept. The schedule above kept {},",
        Param("7", REGISTERED),
        days,
        days,
    )
    doc.definitions(GAP_RULES, reason=REGISTERED)


def instructions(doc: Doc) -> None:
    """The annotator instructions."""
    doc.definitions(INSTRUCTION_GAP, reason=REGISTERED)
    doc.definitions(INSTRUCTION_CASES, reason=CONVENTION)


def round3(doc: Doc, src: Sources) -> None:
    """Section 5b: the registered size, seed, index and exposure of the full relabel."""
    sizes = retest.round3_sizes(src)
    total_questions = sum(s.questions for s in sizes.values())
    total = formatting.count(total_questions)
    doc.check("| {} | Full relabel | Annotation round {} over all {} questions", "12", "3", total)
    doc.check("All {} questions, one annotator, no subsample.", total)
    added = retest.round3_added(src)
    doc.definition("The three questions of deviation D23 add the chunks", reason="deviation D23")
    doc.check(
        "which come to {} chunks on {} questions",
        formatting.count(sum(len(chunks) for chunks in added.values())),
        formatting.count(len(added)),
    )
    doc.ref("The index is `{}`, the one every earlier judgment", corpus.index_version(src))
    rows: dict[str, list[object]] = {
        category: [formatting.count(s.questions), formatting.count(s.decisions)]
        for category, s in sizes.items()
    }
    rows["total"] = [total, formatting.count(sum(s.decisions for s in sizes.values()))]
    doc.table("| Category | Questions | Candidate decisions |", rows)
    doc.check(
        "shuffled per question with seed {}, recorded as `shuffle_seed` in each entry; round 2's",
        formatting.own_form(datasets_rounds.round3_shuffle_seed(src)),
    )
    doc.check(
        "That is {} questions, and on those questions",
        formatting.count(sizes["adversarial"].questions + sizes["version_sensitive"].questions),
    )
    doc.check(
        "showed both verdicts on {} pairs of {} questions, and the D23 relabel showed {} questions",
        formatting.count(retest.adjudicated_pairs(src)),
        formatting.count(len(retest.adjudicated_questions(src))),
        formatting.count(len(retest.relabeled_questions(src))),
    )
    doc.check(
        "\u03ba {} stays the only test-retest figure",
        formatting.metric(retest.registered(src).report.kappa),
    )
    doc.definition(
        "Retrieval: the six methods on `v0.3-gold`", reason="the methods of a retrieval run"
    )
    doc.definition(
        "to pair with the two CPU runs on", reason="the two sensitivity runs of `run_ids`"
    )


def deduplication(doc: Doc, src: Sources) -> None:
    """Section 6: question deduplication and leakage."""
    doc.definitions(DEDUPLICATION_STUDIES, reason=LITERATURE)
    doc.definition("Two checks, both run by", reason=LIST_FOLLOWS)
    doc.definition("differ in one token", reason=CONVENTION)
    doc.definition("a one-token venue swap", reason=CONVENTION)
    threshold = protocol.duplicate_threshold(src)
    doc.check("any pair with cosine \u2265 {} is flagged for manual merge or rewrite", threshold)
    doc.check("{} is expected to lie above the cosine of a one-token venue swap", threshold)
    doc.check("Config key: `eval.duplicate_threshold: {}`", threshold)
    ngram = formatting.count(LEAKAGE_NGRAM)
    doc.check("Any shared n-gram of \u2265 {} tokens that is not an API identifier", ngram)
    doc.check("reaches {} words, after cutting out spans", ngram)
    doc.check(
        "the longest shared run is {} words", formatting.count(gold_sheet.longest_gold_run(src))
    )
    dup = corpus.duplicates(src)
    doc.check(
        "the {} groups of byte-identical chunk text ({} of them Binance mainnet/testnet pairs)",
        formatting.count(dup.groups),
        formatting.count(corpus_sheet.testnet_mirror_groups(src)),
    )

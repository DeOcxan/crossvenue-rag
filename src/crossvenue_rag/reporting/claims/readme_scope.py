"""What README.md claims about its scope: the opening sentence, Status and Limitations."""

from __future__ import annotations

from ... import formatting
from ..check import Doc
from ..compute import answers, calibration, corpus, retrieval
from ..run_ids import ANSWERS_V1
from ..sources import Sources
from .comparisons import comparison_questions, question_count_text

JUDGE_MODEL = "anthropic/claude-opus-5.5"


def scope(doc: Doc, src: Sources) -> None:
    """The opening sentence, the Status list and the Limitations list."""
    judge = src.judged_v1.config.generation.chosen_judge()
    venues = formatting.count(corpus.venue_count(src))
    doc.check("API documentation of {} crypto trading venues (", venues)
    doc.claim(
        "No chunk text occurs on {} venues;",
        formatting.count(corpus.CROSS_VENUE_VENUES),
        holds=corpus.no_cross_venue_texts(src),
        what=f"no chunk text on {formatting.count(corpus.CROSS_VENUE_VENUES)} venues",
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
    doc.check(
        "The sample holds {} answers with {} claims from",
        formatting.count(calibration.answers_rated(src)),
        formatting.count(calibration.sample_claims(src)),
    )
    doc.ref("claims from `results/{}`, on questions of `v0.2.1-gold`", ANSWERS_V1)
    with_human = calibration.judge_with_human(src)
    doc.check(
        "`behaviour_correct` is compared with the human on {} of them and coverage on {}.",
        formatting.count(calibration.compared_count(src, with_human, "behaviour_correct")),
        formatting.count(calibration.compared_count(src, with_human, "coverage")),
    )
    doc.claim(
        f"Opus (`{JUDGE_MODEL}` on OpenRouter)",
        holds=answers.judge_is(src, "opus", JUDGE_MODEL),
        what="the judged run's judge is Opus with this model id",
    )
    doc.check("the full documentation of each of the {} venues", venues)
    doc.check("An LLM drafted all {} questions", question_count_text(src))

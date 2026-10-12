"""What docs/adr/0010-judge.md claims.

The judge candidates, the calibration sample, the answer matrix, the excluded trial
questions and the judges' prices.
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import answers, calibration
from ...sources import Sources
from ..comparisons import question_count_text
from ..reasons import ARXIV_CITATION, OPENROUTER


def _judges(src: Sources) -> str:
    return formatting.count(calibration.judge_candidate_count(src))


def _sample(src: Sources) -> str:
    return formatting.count(calibration.answers_rated(src))


def judge_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0010-judge.md."""
    doc.check(". {} judge candidates, chosen", _judges(src))
    _context(doc, src)
    _decision(doc, src)
    _consequences(doc, src)


def _context(doc: Doc, src: Sources) -> None:
    doc.check(
        "{} questions across {} configurations make {} answers",
        question_count_text(src),
        formatting.count(len(answers.configurations(src))),
        formatting.count(answers.answer_count(src)),
    )
    doc.definition("Rahmani et al. 2025, [arXiv:2506.10301]", reason=ARXIV_CITATION)
    doc.given(
        "is ${} / ${} per million input / output tokens on OpenRouter, "
        "`anthropic/claude-sonnet-5` ${} / ${}",
        "4",
        "20",
        "2",
        "10",
        reason=OPENROUTER,
    )


def _decision(doc: Doc, src: Sources) -> None:
    doc.check("- {} candidates, `anthropic/claude-sonnet-5`", _judges(src))
    doc.given(
        "at most {} output tokens",
        formatting.count(src.default_config.generation.judge_params.max_tokens),
        reason="generation.judge_params.max_tokens",
    )
    doc.check("- Calibration: {} answers of the full generation run", _sample(src))
    doc.check(
        "The {} trial questions are excluded",
        formatting.count(len(src.default_config.generation.calibration_exclude)),
    )
    doc.check("Each candidate grades the same {} answers", _sample(src))


def _consequences(doc: Doc, src: Sources) -> None:
    judges = _judges(src)
    doc.check("measured on {} answers, reaches", _sample(src))
    doc.check(
        "Calibrating {} candidates costs {} gradings of {} answers plus {} repeats",
        judges,
        judges,
        _sample(src),
        judges,
    )

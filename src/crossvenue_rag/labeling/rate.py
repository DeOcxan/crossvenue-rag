"""`crossvenue rate`: a human grades answers in the judge's schema, blind to the judge's grading.

`generation.grader_agreement.compare_gradings` compares the two gradings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from .. import formatting, listing
from ..clock import utc_now
from ..config import Config
from ..corpus.artifacts import chunks_by_id, load_chunks
from ..dataset.questions import load_questions
from ..generation.answer import gold_by_venue
from ..generation.calibration_sample import read_sample
from ..generation.judge_sample import RATINGS_PATH
from ..generation.judging import require_questions, select_answers
from ..generation.records import (
    ANSWERS_FILE,
    AnswerRecord,
    RatingRow,
    append_rating,
    read_answers,
    read_ratings,
)
from ..generation.runs import require_run_index
from ..generation.schemas import Claim, ClaimVerdict, Judgement, VenueCoverage, outcome
from ..index_version import require_current_index
from ..models import Chunk
from .screen import Ask, Show


@dataclass(slots=True)
class RatingView:
    """Everything a rater sees, and nothing else."""

    answer_id: str
    question: str
    venues: list[str]
    expected_behavior: str
    answer: str
    refused: bool
    asked_clarification: bool
    claims: list[Claim]
    cited: dict[int, Chunk]
    gold: dict[str, list[Chunk]]


def rating_view(
    record: AnswerRecord, gold_ids: Sequence[str], chunks: Mapping[str, Chunk]
) -> RatingView:
    """The fields of `record` a rater may see, with the cited and gold chunks.

    Everything the screen renders comes from this view, so none of the judge's
    grading attached to a record reaches the screen. Each claim cites numbers of the
    record's `chunk_ids`, as `read_answers` ensures.

    Raises:
        ValueError: The record's generation failed, or a claim cites a chunk `chunks`
            lacks; the message names it.
    """
    generation = record.generated()
    answer = generation.answer
    numbered = generation.numbered_chunk_ids()
    numbers = sorted({n for claim in answer.claims for n in claim.chunks})
    if lacking := [c for n in numbers if (c := numbered[n]) not in chunks]:
        raise ValueError(f"{record.answer_id} cites {lacking[0]}, which the chunks lack")
    return RatingView(
        answer_id=record.answer_id,
        question=record.question,
        venues=list(record.venues),
        expected_behavior=record.expected_behavior,
        answer=answer.answer,
        refused=outcome(answer) == "refused",
        asked_clarification=answer.asked_clarification,
        claims=list(answer.claims),
        cited={n: chunks[numbered[n]] for n in numbers},
        gold=gold_by_venue(gold_ids, chunks),
    )


def render_rating(show: Show, view: RatingView, position: str) -> None:
    """Prints one answer to rate with everything the rater needs beside it.

    That is the question, its expected behavior, the answer with numbered claims,
    the chunks each claim cites and the gold chunks per venue.
    """
    show(Rule(Text(f"{position}  {view.answer_id}")))
    show(
        Panel(
            Text.assemble(
                (view.question, "bold"),
                "\n\n"
                f"venues: {', '.join(view.venues) or formatting.MISSING}   "
                f"expected: {view.expected_behavior}\n"
                f"refused: {view.refused}   asked for clarification: {view.asked_clarification}",
            ),
            title="question",
            border_style="cyan",
        )
    )
    show(Panel(Text(view.answer), title="answer", border_style="green"))
    for i, claim in enumerate(view.claims, start=1):
        cites = listing.prose([f"[{formatting.count(n)}]" for n in claim.chunks])
        show(Text(f"({formatting.count(i)}) [{claim.venue}] {claim.text}  (cites {cites})"))
    for number, chunk in view.cited.items():
        show(
            Panel(
                Text(chunk.text.strip()),
                title=Text(f"cited [{formatting.count(number)}] {chunk.venue}"),
                border_style="magenta",
            )
        )
    for venue, gold_chunks in sorted(view.gold.items()):
        for i, chunk in enumerate(gold_chunks, start=1):
            show(
                Panel(
                    Text(chunk.text.strip()),
                    title=Text(f"gold {venue} {formatting.count(i)}"),
                    border_style="yellow",
                )
            )


def _yes(prompt: Ask, text: str) -> bool:
    return prompt(Text(text), choices=["y", "n"]) == "y"


def _count(prompt: Ask, text: str, upper: int | None = None) -> int:
    while True:
        raw = prompt(Text(text))
        if raw.isdigit() and (upper is None or int(raw) <= upper):
            return int(raw)


def ask_rating(view: RatingView, prompt: Ask) -> Judgement:
    """A rating in the judge's schema (without rationales).

    Per claim: whether its cited chunks state it and whether it names the right venue.
    Per venue with gold chunks: how many facts the answer needs and how many it states.
    Then whether the behavior was as expected.
    """
    claims = [
        ClaimVerdict(
            index=i,
            supported=_yes(prompt, f"claim {formatting.count(i)}: do its cited chunks state it?"),
            venue_correct=_yes(
                prompt, f"claim {formatting.count(i)}: is it attributed to the right venue?"
            ),
            rationale="",
        )
        for i, _ in enumerate(view.claims, start=1)
    ]
    coverage = []
    for venue in sorted(view.gold):
        gold_facts = _count(
            prompt, f"{venue}: how many facts in its gold chunks does the answer need?"
        )
        covered = _count(prompt, f"{venue}: how many of them does the answer state?", gold_facts)
        coverage.append(
            VenueCoverage(venue=venue, gold_facts=gold_facts, covered_facts=covered, rationale="")
        )
    behaviour = _yes(prompt, f"behavior as expected ({view.expected_behavior})?")
    return Judgement(
        claims=claims,
        coverage=coverage,
        behaviour_correct=behaviour,
        behaviour_rationale="",
    )


def rate_session(
    views: Sequence[RatingView],
    ratings_path: str | Path,
    rater: str,
    show: Show,
    prompt: Ask = Prompt.ask,
    *,
    index_version: str,
) -> int:
    """Rates each view without a saved rating; returns how many were rated.

    Each rating is appended as it is made, with the `index_version` its chunks
    were resolved against, so an interrupt loses nothing already rated.
    """
    path = Path(ratings_path)
    done = {row.answer_id for row in read_ratings(path)}
    todo = [v for v in views if v.answer_id not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    rated = 0
    try:
        for i, view in enumerate(todo, start=1):
            render_rating(show, view, f"{formatting.count(i)}/{formatting.count(len(todo))}")
            rating = ask_rating(view, prompt)
            row = RatingRow(
                answer_id=view.answer_id,
                rater=rater,
                at=utc_now(),
                rating=rating,
                index_version=index_version,
            )
            append_rating(path, row)
            rated += 1
    except KeyboardInterrupt:
        show(Text("\ninterrupted: ratings so far are saved", style="yellow"))
    return rated


def rate_calibration_sample(
    cfg: Config, sample_file: Path, questions_path: str | Path, rater: str, show: Show
) -> int:
    """Rates the answers `sample_file` names, by hand; returns how many were rated.

    The ratings go to `RATINGS_PATH` with the `index_version` the run was made against.
    """
    run_dir, ids = read_sample(sample_file)
    version = require_run_index(Path(run_dir), require_current_index(cfg))
    chosen = select_answers(read_answers(Path(run_dir) / ANSWERS_FILE), ids)
    by_question = require_questions(chosen, load_questions(questions_path))
    chunks = chunks_by_id(load_chunks(cfg))
    views = [rating_view(r, by_question[r.question_id].gold_chunk_ids, chunks) for r in chosen]
    return rate_session(views, RATINGS_PATH, rater, show, index_version=version)

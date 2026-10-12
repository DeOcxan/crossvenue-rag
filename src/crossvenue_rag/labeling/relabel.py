"""`crossvenue label --relabel ID ...`: judging named questions again under the amended criterion.

The normal label screen shows each question's recorded round-1 candidates plus the
chunks a file adds (`--add-chunks`), for questions whose earlier labels a review found
wrong. The verdict goes to the journal under the key `relabeled`; the question file,
round 1 and round 2 are not written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from .. import formatting, listing
from ..clock import utc_now
from ..dataset.journal import RELABELED, Judgment, Relabel, load_rounds, rounds_path, save_rounds
from ..dataset.questions import load_questions
from ..errors import UserError
from ..models import Chunk, Result
from .adjudicate import CRITERION_DATE, CRITERION_SUMMARY
from .judge import judge
from .screen import Ask


def relabel_candidates(
    chunks: Mapping[str, Chunk],
    question_id: str,
    round1_entry: Judgment,
    added: Sequence[str],
) -> list[Result]:
    """Round 1's judged chunks in their recorded order, then the added ones it never showed.

    Raises:
        UserError: a chunk is not in the corpus.
    """
    ids = round1_entry.judged_ids()
    ids += [c for c in dict.fromkeys(added) if c not in ids]
    missing = [c for c in ids if c not in chunks]
    if missing:
        raise UserError(
            f"{question_id}: {formatting.plural(len(missing), 'chunk')} not in the corpus: "
            f"{listing.first_items(missing)}",
            hint="the chunk ids are of another index; carry them with `remap-ids`",
        )
    return [Result.from_chunk(chunks[c], 0.0) for c in ids]


@dataclass(frozen=True)
class RelabelTarget:
    """The questions to relabel, the chunks each adds, and what every verdict is stamped with."""

    questions_path: str
    ids: Sequence[str]
    added: Mapping[str, Sequence[str]]
    chunks: Mapping[str, Chunk]
    index_version: str
    annotator: str


def relabel_session(
    target: RelabelTarget, *, console: Console | None = None, prompt: Ask = Prompt.ask
) -> None:
    """Shows each named question on the label screen and journals the verdict under `relabeled`.

    Ctrl-C stops; verdicts given so far are saved, and a rerun shows every named question.

    Raises:
        UserError: a named question is not in the file or has no round-1 entry.
    """
    console = console or Console()
    path = target.questions_path
    rounds = load_rounds(path)
    questions = {q.id: q for q in load_questions(path)}
    unknown = [q for q in target.ids if q not in questions or rounds.first(q) is None]
    if unknown:
        raise UserError(
            f"no round-1 judgment to start from: {' '.join(unknown)}",
            hint=f"relabel questions round 1 judged; the journal is {rounds_path(path)}",
        )
    console.print(Text(CRITERION_SUMMARY, style="dim"))
    done = 0
    try:
        for qid in target.ids:
            question, first = questions[qid], rounds.require_first(qid)
            candidates = relabel_candidates(target.chunks, qid, first, target.added.get(qid, ()))
            console.rule(Text(qid))
            judgement = judge(console, question, candidates, set(), prompt)
            if judgement is None:
                continue
            rounds.entry(qid).relabeled = Relabel(
                annotator=target.annotator,
                annotated_at=utc_now(),
                criterion=CRITERION_DATE,
                index_version=target.index_version,
                pool=[r.chunk_id for r in candidates],
                relevant=judgement.relevant,
                venue_coverage=judgement.venue_coverage or None,
            )
            save_rounds(path, rounds)
            done += 1
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted: verdicts so far are saved[/yellow]")
    console.print(
        Text(
            f"{formatting.count(done)}/{formatting.count(len(target.ids))} questions recorded "
            f"under '{RELABELED}'. Journal: {rounds_path(path)}"
        )
    )

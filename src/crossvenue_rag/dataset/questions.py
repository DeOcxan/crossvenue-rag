"""The evaluation set on disk: the question file, checked against the index its labels used.

`<name>.jsonl` holds the questions, one JSON object per line, and round 1 writes gold into it.
The judgment journal beside it is `dataset.journal`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .. import formatting, jsonl, listing
from ..errors import UserError, require_file
from ..models import Question, missing_fields, wrongly_typed_fields
from ..retrieval.retriever import IndexCheck
from ..run_files import read_json_model
from .journal import FIRST_ROUND, Journal, load_rounds, rounds_path
from .records import DatasetMeta

# Below this many questions a per-category metric is not worth reporting, and selection
# takes no category below it (evals/methodology.md on set size and power, decision 2).
MIN_QUESTIONS_PER_CATEGORY = 5


# What a line that is not a question is told to do.
_LINE_HINT = "fix that line of the file"
# What a dataset whose `.meta.json` cannot be read is told to do.
META_HINT = "restore the .meta.json the dataset version was published with"


def load_questions(path: str | Path) -> list[Question]:
    """The questions in a question file, in file order.

    Fields `Question` does not have are ignored, as a dataset file carries more.

    Raises:
        UserError: Nothing is at `path`, or a directory is. A field has the wrong type or value, a
            required field is missing, or an id repeats. The error names the line.
    """
    path = require_file(path, hint="check the path given by --questions or eval.questions")
    questions: list[Question] = []
    first_line: dict[str, int] = {}
    for number, row in jsonl.iter_rows(path):
        if wrong := wrongly_typed_fields(row, Question):
            problems = "; ".join(f"{name}: {problem}" for name, problem in wrong.items())
            raise UserError(f"{formatting.location(path, number)}: {problems}", hint=_LINE_HINT)
        missing = missing_fields(row, Question)
        if missing:
            raise UserError(
                f"{formatting.location(path, number)}: the record has no {listing.prose(missing)}",
                hint=_LINE_HINT,
            )
        question = Question.from_dict(row)
        if question.id in first_line:
            raise UserError(
                f"{formatting.location(path, number)}: id {question.id} is also on line "
                f"{formatting.own_form(first_line[question.id])}",
                hint="give every question its own id",
            )
        first_line[question.id] = number
        questions.append(question)
    return questions


def save_questions(path: str | Path, questions: Sequence[Question]) -> None:
    """Writes the questions to a question file, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write(path, (q.to_dict() for q in questions))


def label_version_problems(
    questions: Sequence[Question], journal: Journal, current: str, *, all_rounds: bool = False
) -> list[str]:
    """Labeled questions whose judgments were not made against `current`.

    A labeled question is one with gold chunks or an annotation timestamp. Its
    round-1 judgment in the sidecar carries the `index_version` it was made
    against; an entry without one counts as a mismatch.
    `all_rounds` also checks later rounds, which `label` needs (a round-2 pool
    from a different index is not comparable with round 1) and `eval` does not.
    """
    problems: list[str] = []
    for question in questions:
        entries = journal.get(question.id)
        first = entries.first
        if all_rounds:
            checked = entries.entries()
        elif question.is_judged:
            checked = [(str(FIRST_ROUND), first.index_version if first is not None else None)]
        else:
            checked = []
        for round_id, version in checked:
            if version != current:
                problems.append(
                    f"{question.id} round {round_id}: "
                    + (f"labeled against {version}" if version else "no index_version recorded")
                )
    return problems


def require_label_versions(
    questions_path: str | Path, current: str, *, all_rounds: bool = False
) -> None:
    """Refuses to go on if any label was made against a different index.

    A frozen dataset (`evals/datasets/<name>.jsonl`) is checked against the `index_version`
    its `.meta.json` records, the index every label in it was judged against, checked at
    freeze time; the copy of its journal taken then is not read here.

    Raises:
        UserError: A frozen dataset was frozen against another index, or a judgment in the
            journal was made against one.
    """
    frozen = frozen_index_version(questions_path)
    if frozen is not None:
        if frozen != current:
            raise UserError(
                f"{questions_path} was frozen against index {frozen}, the current index is "
                f"{current}",
                hint="a frozen dataset cannot be relabeled in place: migrate its ids into "
                "another version (`crossvenue id-map` + `remap-ids`) or relabel and freeze it "
                "again",
            )
        return
    problems = label_version_problems(
        load_questions(questions_path), load_rounds(questions_path), current, all_rounds=all_rounds
    )
    if not problems:
        return
    shown = listing.first_items(problems, "; ")
    raise UserError(
        f"{formatting.plural(len(problems), 'judgment')} in {rounds_path(questions_path)} "
        "were not made against "
        f"the current index {current}: {shown}",
        hint="their chunk ids may point at different text, or at nothing. If the index "
        "changed only by renaming ids (as deduplication did), build a map with `crossvenue "
        "id-map` and apply it with `crossvenue remap-ids`, which also restamps the sidecar; "
        "otherwise re-judge them with `crossvenue label --all`.",
    )


def require_gold_resolves(questions: Sequence[Question], chunk_ids: set[str]) -> None:
    """Refuses to score against gold ids the current corpus does not contain.

    The `index_version` check covers the recorded inputs of the corpus. A gold id
    absent from the corpus means the chunks changed anyway (ADR 0004), and
    a metric computed against it would count a missing chunk as a miss.

    Raises:
        UserError: A gold chunk id of `questions` is not in `chunk_ids`.
    """
    missing = [
        (q.id, chunk_id)
        for q in questions
        for chunk_id in q.gold_chunk_ids
        if chunk_id not in chunk_ids
    ]
    if not missing:
        return
    shown = listing.first_items([f"{qid}:{cid}" for qid, cid in missing])
    raise UserError(
        f"{formatting.plural(len(missing), 'gold chunk id')} not in the current corpus: {shown}",
        hint="rebuild the corpus from the manifest (`make fetch ingest`) or migrate the ids "
        "with `crossvenue id-map` + `remap-ids` into another dataset version",
    )


def index_check(questions_path: str | Path, questions: Sequence[Question] = ()) -> IndexCheck:
    """The check that a question file may be used at an index (`retrieval.retriever.IndexCheck`).

    It passes when every label of the file was made at the index's `index_version`
    (`require_label_versions`) and every gold id of `questions` is one of the index's
    chunks (`require_gold_resolves`); no questions, no gold check.
    """

    def check(version: str, chunk_ids: set[str]) -> None:
        require_label_versions(questions_path, version)
        require_gold_resolves(questions, chunk_ids)

    return check


def meta_path(questions_path: str | Path) -> Path:
    """The `.meta.json` beside a question file: its name with the last suffix replaced."""
    return Path(questions_path).with_suffix(".meta.json")


def frozen_index_version(questions_path: str | Path) -> str | None:
    """`index_version` recorded in a dataset's `.meta.json`, if it has one.

    Raises:
        UserError: The `.meta.json` is not UTF-8 or does not validate as a dataset's
            metadata.
    """
    meta = meta_path(questions_path)
    if not meta.exists():
        return None
    return read_json_model(meta, DatasetMeta, hint=META_HINT).corpus_snapshot.index_version

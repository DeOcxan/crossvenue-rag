"""The never-judged chunks an augmentation shows, listed from one run.

The run is an evaluation run or a retrieval-only pool-construction run from `pool_run`;
the list is tied to that run and its index, and `labeling.augment` shows exactly those chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .. import formatting, jsonl
from ..config import Config
from ..dataset.journal import Journal
from ..dataset.questions import load_questions
from ..errors import UserError, require_file
from ..evaluation.results import PER_QUERY_FILE, PerQueryRow, read_per_query
from ..models import Question
from ..retrieval.methods import method_named
from ..retrieval.protocols import Ranks
from ..run_files import question_fields, read_index, write_csv, write_index, write_run_config


class Candidate(BaseModel):
    """A never-judged chunk, with the rank each method that found it gave it."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    found_by: dict[str, int]


class CandidateList(BaseModel):
    """One question's never-judged chunks from a run, best rank first, with the run's stamps."""

    model_config = ConfigDict(extra="forbid")

    id: str
    chunks: list[Candidate]
    run: str
    methods: list[str]
    k: int
    index_version: str


@dataclass(frozen=True)
class RunRankings:
    """A run's ranked lists as `augment_candidates` reads them.

    The run directory, the index its `index.json` names, and the rows of its
    `per_query.csv`.
    """

    run: str
    index_version: str
    rows: list[PerQueryRow]


def read_run_rankings(run_dir: str | Path) -> RunRankings:
    """The ranked lists of an evaluation or pool-construction run.

    Raises:
        UserError: The run has no `index.json`, or no `per_query.csv`; an `index.json` that
            is not one is reported by `run_files.read_index`.
    """
    run_dir = Path(run_dir)
    hint = "pass the directory of a `crossvenue eval` or `crossvenue pool-run` run"
    index = read_index(run_dir)
    if index is None:
        raise UserError(f"{run_dir} has no index.json naming its index", hint=hint)
    require_file(run_dir / PER_QUERY_FILE, hint=hint)
    return RunRankings(str(run_dir), index.index_version, list(read_per_query(run_dir).rows))


def augment_candidates(
    rankings: RunRankings,
    questions: Sequence[Question],
    journal: Journal,
    methods: Sequence[str],
    category: str | None,
    k: int,
) -> list[CandidateList]:
    """Per question, the chunks `methods` ranked in the top k that were never judged.

    Chunks are ordered by the best rank any of the methods gave them, then by id,
    so the likeliest candidates come first. Questions with no unjudged chunk are
    left out. Only questions with a round-1 judgment are considered.
    """
    by_id = {q.id: q for q in questions}
    found: dict[str, dict[str, dict[str, int]]] = {}
    for row in rankings.rows:
        question = by_id.get(row.question_id)
        if question is None or row.method not in methods:
            continue
        if category is not None and question.effective_category != category:
            continue
        entry = journal.first(question.id)
        if entry is None:
            continue
        judged = set(entry.judged_ids())
        for rank, chunk_id in enumerate(row.ids[:k], start=1):
            if chunk_id not in judged:
                found.setdefault(question.id, {}).setdefault(chunk_id, {})[row.method] = rank

    out: list[CandidateList] = []
    for question in questions:
        chunks = found.get(question.id)
        if not chunks:
            continue
        ordered = sorted(chunks.items(), key=lambda kv: (min(kv[1].values()), kv[0]))
        out.append(
            CandidateList(
                id=question.id,
                chunks=[Candidate(chunk_id=c, found_by=ranks) for c, ranks in ordered],
                run=rankings.run,
                methods=list(methods),
                k=k,
                index_version=rankings.index_version,
            )
        )
    return out


# Saved in the run's index.json; its wording stays as committed runs have it
# (docs/code-style.md rule 26). POOL_RUN.md states the same in prose.
POOL_RUN_PURPOSE = (
    "pool construction only: retrieval depth for building augmentation candidates. "
    "Not an evaluation -- no metrics are computed and none may be quoted from it."
)
POOL_RUN_NOTE = (
    "Pool construction only: retrieval depth for building augmentation candidates. "
    "It is not an evaluation: no metrics are computed and none may be quoted from it."
)


@dataclass(frozen=True)
class PoolSpec:
    """What a pool-construction run retrieves.

    The question file (or one category of it), the methods, and the depth k.
    """

    questions_path: str | Path
    methods: Sequence[str]
    k: int
    category: str | None = None


def _ranked_ids(retriever: Ranks, question: Question, method: str, k: int) -> str:
    """A method's top k for a question, as space-separated chunk ids."""
    results = retriever.retrieve(
        question.question, method=method_named(method), k=k, venues_hint=question.venues
    )
    return " ".join(r.chunk_id for r in results)


def pool_run(
    cfg: Config, retriever: Ranks, spec: PoolSpec, index_version: str, run_dir: Path
) -> int:
    """Retrieves the top k of `spec.methods` for each question and writes only that.

    The output has the shape `augment_candidates` reads (`per_query.csv` with
    `retrieved_ids`, `index.json`), plus `purpose` in index.json and a
    `POOL_RUN.md` saying it is not an evaluation. There is deliberately no
    summary, report or metric: a run built to find chunks to judge is not a run
    to score methods on.
    """
    questions_path, methods, k, category = spec.questions_path, spec.methods, spec.k, spec.category
    questions = [
        q
        for q in load_questions(questions_path)
        if category is None or q.effective_category == category
    ]
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        [question.id, method, k, _ranked_ids(retriever, question, method, k)]
        for method in methods
        for question in questions
    ]
    write_csv(run_dir / PER_QUERY_FILE, ["question_id", "method", "k", "retrieved_ids"], rows)
    source = Path(questions_path)
    index = {
        "index_version": index_version,
        "purpose": POOL_RUN_PURPOSE,
        **question_fields(source),
        "methods": list(methods),
        "k": k,
        "category": category,
        "questions_run": len(questions),
    }
    write_index(run_dir, index, final_newline=False)
    write_run_config(run_dir, cfg)
    (run_dir / "POOL_RUN.md").write_text(
        f"# Pool-construction run\n\n{POOL_RUN_NOTE}\n\n"
        f"- methods: {', '.join(methods)}\n- k: {formatting.count(k)}\n"
        f"- category: {category or 'all'}\n"
        f"- questions: {formatting.count(len(questions))} from `{source}`\n"
        f"- `index_version`: `{index_version}`\n",
        encoding="utf-8",
    )
    return len(rows)


def save_candidates(path: str | Path, entries: Sequence[CandidateList]) -> None:
    """Writes the candidate lists to a JSONL file, one per line."""
    jsonl.write(path, (e.model_dump() for e in entries))


def load_candidates(path: str | Path) -> list[CandidateList]:
    """The candidate lists of a JSONL file, in file order.

    Raises:
        UserError: A line is not a candidate list; the message names the line.
    """
    return jsonl.read_models(
        path, CandidateList, hint="rebuild the list with `crossvenue augment-candidates`"
    )

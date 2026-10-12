"""The evaluation run as `crossvenue eval` starts it.

`run_evaluation` runs every method over every question with a checked retriever and
writes `results/<ts>/`. The checks of the question file belong to `dataset`, a layer
this one does not import, so the command passes them in (`EvalJob.check`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path

from ..config import Config
from ..config_pipeline import RERANKED_METHODS
from ..corpus.artifacts import venue_chunk_counts
from ..errors import UserError
from ..models import Question
from ..retrieval.retriever import IndexCheck, Retriever, checked_retriever
from .evaluator import Evaluator, summarise
from .results import run_tables, write_eval_run


@dataclass(frozen=True)
class EvalJob:
    """The `eval` inputs: the question file, its questions and round-1 judged sets, the checks."""

    questions_path: Path
    questions: Sequence[Question]
    judged: dict[str, set[str]] | None
    check: IndexCheck
    _: KW_ONLY
    venue_filter: bool


def run_evaluation(
    cfg: Config,
    job: EvalJob,
    *,
    open_retriever: Callable[[Config], Retriever] | None = None,
) -> Path:
    """The directory of the run the ablation over `job.questions` writes.

    The resolved config written into the run names `job.questions_path`, the file
    that ran, even when an override replaced the default. `open_retriever` builds
    the retriever from `cfg` (`checked_retriever`).

    Raises:
        UserError: The question file holds no question (no run directory is written).
    """
    if not job.questions:
        raise UserError(
            f"{job.questions_path} holds no question, so there is nothing to evaluate",
            hint="pass --questions with a question file that has questions",
        )
    cfg = cfg.with_questions(job.questions_path)
    reranks = any(m in RERANKED_METHODS for m in cfg.eval.methods)
    with checked_retriever(
        cfg, job.check, load_reranker=reranks, open_retriever=open_retriever
    ) as (retriever, version):
        runs = Evaluator(cfg, retriever).run(
            job.questions, filter_by_question_venues=job.venue_filter, judged=job.judged
        )
    summaries = summarise(runs, cfg.eval.k_values)
    tables = run_tables(cfg, runs, summaries, job.questions, venue_chunk_counts(cfg))
    tables = replace(tables, venue_filter=job.venue_filter)
    return write_eval_run(cfg, tables, job.questions, index_version=version)

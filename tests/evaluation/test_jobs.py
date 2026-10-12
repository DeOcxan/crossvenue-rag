"""Tests for `evaluation.jobs`: the eval run on the fixture corpus with an in-memory retriever."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.corpus.artifacts import load_chunks
from crossvenue_rag.errors import UserError
from crossvenue_rag.evaluation import jobs
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.index_version import require_current_index, write_index_meta
from crossvenue_rag.models import Question
from tests.conftest import InMemoryRetriever, build_corpus


class MemoryRetriever(InMemoryRetriever):
    """`Retriever` as the job opens it, over the fixture corpus: no model, no Postgres."""

    def __init__(self, cfg: Config) -> None:
        chunks = load_chunks(cfg)
        super().__init__(cfg, chunks, BM25Index.build(chunks))

    warmups: ClassVar[list[bool]] = []

    def warmup(self, *, load_reranker: bool = True) -> None:
        MemoryRetriever.warmups.append(load_reranker)

    def require_current_index(self) -> str:
        return require_current_index(self.cfg)


@pytest.fixture
def indexed(cfg):
    write_index_meta(cfg, len(build_corpus(cfg)))
    return cfg


def questions(cfg):
    uniswap = next(c.chunk_id for c in load_chunks(cfg) if c.venue == "uniswap")
    return [Question("cv-0001", "uniswap fee tiers", "gold", ["uniswap"], "fact", [uniswap],
                     category="fact")]  # fmt: skip


def test_the_run_is_checked_evaluated_and_written(indexed, tmp_path):
    checked = []
    path = tmp_path / "questions.jsonl"
    job = jobs.EvalJob(
        path, questions(indexed), None, lambda v, ids: checked.append((v, ids)), venue_filter=False
    )
    run_dir = jobs.run_evaluation(indexed, job, open_retriever=MemoryRetriever)
    version, chunk_ids = checked[0]
    assert version == require_current_index(indexed)
    assert chunk_ids == {c.chunk_id for c in load_chunks(indexed)}
    assert (run_dir / "per_query.csv").exists()
    # The resolved config names the question file that ran.
    assert str(path) in (run_dir / "config.yaml").read_text().replace("\n", "")


def test_a_failed_check_writes_no_run(indexed, tmp_path):
    def refuse(version: str, chunk_ids: set[str]) -> None:
        raise UserError("labelled against another index")

    job = jobs.EvalJob(Path("q.jsonl"), questions(indexed), None, refuse, venue_filter=False)
    with pytest.raises(UserError, match="another index"):
        jobs.run_evaluation(indexed, job, open_retriever=MemoryRetriever)
    assert not indexed.paths.results.exists() or not any(indexed.paths.results.iterdir())


@pytest.mark.parametrize(
    ("methods", "loaded"), [(["bm25", "venue_split"], True), (["bm25", "dense"], False)]
)
def test_the_warmup_loads_the_reranker_for_any_reranking_method(indexed, tmp_path, methods, loaded):
    indexed.eval.methods = methods
    job = jobs.EvalJob(
        tmp_path / "q.jsonl", questions(indexed), None, lambda v, ids: None, venue_filter=False
    )
    jobs.run_evaluation(indexed, job, open_retriever=MemoryRetriever)
    assert MemoryRetriever.warmups[-1] is loaded


def test_an_empty_question_file_is_refused_before_a_run_is_written(indexed, tmp_path):
    job = jobs.EvalJob(tmp_path / "empty.jsonl", [], None, lambda v, ids: None, venue_filter=False)
    with pytest.raises(UserError, match="holds no question") as refused:
        jobs.run_evaluation(indexed, job, open_retriever=MemoryRetriever)
    assert "--questions" in (refused.value.hint or "")
    assert not indexed.paths.results.exists() or not any(indexed.paths.results.iterdir())


def test_the_venue_filter_is_passed_by_keyword_only(tmp_path):
    with pytest.raises(TypeError):
        # The positional boolean is the call under test.
        jobs.EvalJob(tmp_path / "q.jsonl", [], None, lambda v, ids: None, True)  # type: ignore[call-arg]  # noqa: FBT003
    job = jobs.EvalJob(tmp_path / "q.jsonl", [], None, lambda v, ids: None, venue_filter=True)
    assert job.venue_filter

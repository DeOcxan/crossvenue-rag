"""Shared helpers for tests/labeling/."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from crossvenue_rag.config import Config
from crossvenue_rag.config_pipeline import RetrievalMethod
from crossvenue_rag.dataset.journal import load_rounds
from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.labeling.candidates import (
    augment_candidates,
    read_run_rankings,
    save_candidates,
)
from crossvenue_rag.models import Chunk, ChunkSource, Question, Result, VenueCoverage
from tests.conftest import write_journal


def result(chunk_id: str, text: str, copies: int = 0) -> Result:
    sources = [ChunkSource("binance", "rest-api.md", ["Limits"], "", 0, environment="mainnet")]
    sources += [
        ChunkSource("binance", f"testnet/rest-api-{i}.md", ["Limits"], "", 0, environment="testnet")
        for i in range(copies)
    ]
    return Result(
        chunk_id=chunk_id,
        score=0.0,
        text=text,
        venue="binance",
        venue_type="cex",
        doc_path="rest-api.md",
        heading_path=["Limits"],
        source_url="",
        sources=sources,
    )


def quiet() -> Console:
    return Console(record=True, width=160)


def comparison(**kw: Any) -> Question:
    return Question(
        kw.pop("id", "cv-0001"),
        "how do binance and bybit differ here",
        "silver",
        ["binance", "bybit"],
        "comparison",
        category="comparison",
        **kw,
    )


def results() -> list[Result]:
    return [Result.from_chunk(c, 1.0) for c in CHUNKS]


def chunk(chunk_id: str, venue: str, doc_path: str = "doc.md", env: str | None = None) -> Chunk:
    source = ChunkSource(venue, doc_path, ["H"], "", 0, environment=env)
    return Chunk(
        chunk_id=chunk_id,
        venue=venue,
        venue_type="cex",
        doc_path=doc_path,
        heading_path=["H"],
        source_url="",
        text=f"H\n\n{venue} text of {chunk_id}.\nmore.",
        token_count=5,
        position=0,
        sources=[source],
        environment=env,
    )


def setup_augment(tmp_path: Path, bybit: VenueCoverage = "missing_from_pool") -> tuple[str, Path]:
    path = tmp_path / "questions.jsonl"
    question = Question(
        "cv-0001",
        "how do binance and bybit differ",
        "silver",
        ["binance", "bybit"],
        "comparison",
        category="comparison",
        gold_chunk_ids=["g1"],
        venue_coverage={"binance": "covered", "bybit": bybit},
        annotated_at="2026-09-24T21:00:00Z",
    )
    save_questions(path, [question])
    write_journal(
        path,
        {
            "cv-0001": {
                "1": {
                    "annotator": "me",
                    "annotated_at": "2026-09-24T21:00:00Z",
                    "index_version": VERSION,
                    "pool": ["g1", "p1"],
                    "relevant": ["g1"],
                    "venue_coverage": {"binance": "covered", "bybit": bybit},
                }
            }
        },
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "index.json").write_text(json.dumps({"index_version": VERSION}))
    with (run / "per_query.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question_id", "method", "retrieved_ids"])
        writer.writerow(["cv-0001", "hybrid_rerank", "g1 p1"])
        writer.writerow(["cv-0001", "venue_split", "g1 n1 p1 n2"])
        writer.writerow(["cv-0001", "venue_split_oracle", "n2 g1"])
    candidates = tmp_path / "augment.jsonl"
    entries = augment_candidates(
        read_run_rankings(run),
        load_questions(path),
        load_rounds(path),
        ["venue_split", "venue_split_oracle"],
        "comparison",
        10,
    )
    save_candidates(candidates, entries)
    return str(path), candidates


def aug_chunk(chunk_id: str, venue: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        venue=venue,
        venue_type="cex",
        doc_path=f"{chunk_id}.md",
        heading_path=["H"],
        source_url="",
        text=f"H\n\n{venue} text of {chunk_id}.\nmore.",
        token_count=5,
        position=0,
        sources=[ChunkSource(venue, f"{chunk_id}.md", ["H"], "", 0)],
    )


CHUNKS = [
    chunk("c1", "binance", "rest-api.md", env="mainnet"),
    chunk("c2", "bybit", "docs/v5/x.mdx"),
    chunk("c3", "binance", "testnet/rest-api.md", env="testnet"),
]


VERSION = "idx-test"


AUG_CHUNKS = {
    c.chunk_id: c
    for c in [
        aug_chunk("g1", "binance"),  # current gold
        aug_chunk("p1", "bybit"),  # in the round-1 pool, judged irrelevant
        aug_chunk("n1", "bybit"),  # unjudged: only venue_split found it
        aug_chunk("n2", "binance"),  # unjudged
    ]
}


class Script:
    """Prompt stand-in that replays answers and fails loudly when it runs out."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def __call__(self, message: str, **_: object) -> str:
        self.asked.append(message)
        if not self.answers:
            raise AssertionError(f"unexpected prompt: {message}")
        return self.answers.pop(0)


class ThreeChunkRetriever:
    """Every method returns the three chunks in order; ids resolve via by_id."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.by_id = {c.chunk_id: c for c in CHUNKS}

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int | None = None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        return [Result.from_chunk(c, 1.0) for c in CHUNKS][:k]


class ByIdRetriever:
    by_id = AUG_CHUNKS


class RankingRetriever(ByIdRetriever):
    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int | None = None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        order = ["g1", "p1", "n1", "n2"] if method == "venue_split" else ["n2", "g1"]
        return [Result.from_chunk(AUG_CHUNKS[c], 1.0) for c in order][:k]

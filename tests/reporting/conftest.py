"""Fixture artifacts for the reporting tests: two answer runs, a gold set, a
retrieval run, a corpus and plain files, each written under a tmp root."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from crossvenue_rag import jsonl, run_files
from crossvenue_rag.config import Config
from crossvenue_rag.config_pipeline import PathsConfig
from crossvenue_rag.corpus.artifacts import save_chunks
from crossvenue_rag.generation.answer_rows import SCOPES, mean_column
from crossvenue_rag.generation.metrics import METRICS
from crossvenue_rag.models import Chunk
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, ANSWERS_V2, GOLD, JUDGED_V1
from tests.generation.factories import answer_json, generation_json, record_json

# The answer configurations of the fixture runs, as their generation.matrix lists them.
CONFIGS = ("gemini@venue_split", "qwen@venue_split", "mercury@venue_split", "gemini@hybrid_rerank")
MATRIX = [{"name": c, "generator": c.split("@")[0], "retriever": c.split("@")[1]} for c in CONFIGS]
# The config.yaml of a fixture answer or judged run.
ANSWER_CONFIG = {
    "generation": {
        "generators": {g: {"model": f"vendor/{g}"} for g in ("gemini", "qwen", "mercury")},
        "matrix": MATRIX,
        "primary_pair": [CONFIGS[0], CONFIGS[3]],
    }
}


def answer_record(
    *, refused: bool = False, partial: bool = False, claims: int = 1, asked: bool = False
) -> dict[str, Any]:
    """A generated answer's JSON, as `answer_json` builds it."""
    return answer_json(refused=refused, partial=partial, claims=claims, asked=asked)


def write_run(root: Path, run_id: str, answers: dict[tuple[str, str], dict[str, Any]]) -> None:
    """answers: {(config, question id): answer}; every call costs 0.01."""
    directory = root / "results" / run_id
    directory.mkdir(parents=True)
    records = [
        record_json(
            config,
            qid,
            category="comparison" if qid == "cv-0001" else "unanswerable",
            generation=generation_json(answer=answer, chunk_ids=("c1",)),
        )
        for (config, qid), answer in answers.items()
    ]
    write_jsonl(directory / "answers.jsonl", records)
    (directory / "config.yaml").write_text(yaml.safe_dump(ANSWER_CONFIG), encoding="utf-8")


def write_fixture(root: Path) -> None:
    (root / "evals/datasets").mkdir(parents=True)
    questions: list[dict[str, Any]] = [
        {"id": "cv-0001", "category": "comparison", "expected_behavior": "answer"},
        {"id": "cv-0002", "category": "unanswerable", "expected_behavior": "refuse"},
    ]
    for q in questions:
        q.update(
            question="q",
            tier="gold",
            qtype="comparison",
            venues=["binance", "bybit"],
            gold_chunk_ids=[],
        )
    jsonl.write(root / GOLD, questions)
    v1 = {(c, q): answer_record() for c in CONFIGS for q in ("cv-0001", "cv-0002")}
    v1[("gemini@venue_split", "cv-0001")] = answer_record(refused=True, claims=0)
    v2 = dict(v1)
    v2[("gemini@venue_split", "cv-0001")] = answer_record(partial=True)
    # Flagged partial, no claim: a refusal under the outcome rule.
    v2[("mercury@venue_split", "cv-0001")] = answer_record(partial=True, claims=0)
    write_run(root, ANSWERS_V1, v1)
    write_run(root, ANSWERS_V2, v2)
    # The judged run's config pre-registers the primary pair.
    write_file(root, f"results/{JUDGED_V1}/config.yaml", yaml.safe_dump(ANSWER_CONFIG))


def write_file(root: Path, relative: str, text: str) -> None:
    """A text file at `relative` under `root`, its directories created."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_chunks(root: Path, chunks: list[Chunk]) -> None:
    """`chunks` (a list of `models.Chunk`) as data/processed/chunks.jsonl under `root`."""
    save_chunks(Config(paths=PathsConfig(processed=root / "data/processed")), chunks)


# The columns each saved table's model requires, with a neutral value, for
# fixtures that state only the columns a test reads.
PAIRED_ESTIMATE = {"n": 0, "mean_a": 0.0, "mean_b": 0.0, "diff": 0.0, "low": 0.0, "high": 0.0}
REQUIRED: dict[str, dict[str, Any]] = {
    "summary": {"questions": 0, "labelled_questions": 0, "p50_ms": 0.0, "p95_ms": 0.0},
    "by_category": {"questions": 0, "labelled_questions": 0, "p50_ms": 0.0},
    "per_query": {"method": "bm25", "retrieved_ids": ""},
    "paired": {**PAIRED_ESTIMATE, "better": 0, "worse": 0, "ties": 0, "mcnemar_p": ""},
    "answer_paired": {
        **PAIRED_ESTIMATE,
        "randomisation_p": "",
        "mcnemar_p": "",
        "role": "secondary",
    },
    "answer_summary": {
        **dict.fromkeys(("answers", "errors", "refused", "asked_clarification", "partial"), 0),
        **dict.fromkeys(
            (
                "input_tokens",
                "reasoning_tokens",
                "output_tokens",
                "cost_usd",
                "cost_usd_new",
                "judge_input_tokens",
                "judge_reasoning_tokens",
                "judge_output_tokens",
                "judge_cost_usd",
                "http_retries",
                "invalid_responses",
                "judge_invalid_responses",
            ),
            0.0,
        ),
        "latency_p50_ms": "",
        "latency_p95_ms": "",
        "cached_calls": 0,
        "params": "",
        **dict.fromkeys((mean_column(m, s) for m in METRICS for s in SCOPES), ""),
    },
}


def complete(kind: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """`rows` with each column `REQUIRED[kind]` names and a row lacks set to its neutral value."""
    return [{**REQUIRED[kind], **row} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """`rows` (dicts, one header for all rows) as a CSV file at `path`, its directory created.

    The header is the first row's keys; no row writes an empty file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0])
    run_files.write_csv(path, header, ([row[h] for h in header] for row in rows))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """`rows` as a JSONL file at `path`, its directory created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write(path, rows)


def write_retrieval_run(
    root: Path, run_id: str, *, config: dict[str, Any] | None = None, **tables: list[dict[str, Any]]
) -> None:
    """A retrieval run under results/: config.yaml, and a CSV file per table
    (`summary`, `by_category`, `per_query`, `paired`) from its list of rows."""
    directory = root / "results" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(config or {}))
    for name, rows in tables.items():
        if rows:
            write_csv(directory / f"{name}.csv", complete(name, rows))


def write_gold(root: Path, questions: list[dict[str, Any]], path: str = GOLD) -> None:
    """A gold set at `path`: each question a dict with at least `id`, the fields a
    `models.Question` needs filled with a two-venue comparison question."""
    defaults = {
        "question": "q",
        "tier": "gold",
        "venues": ["binance", "bybit"],
        "qtype": "comparison",
        "category": "comparison",
        "gold_chunk_ids": [],
    }
    write_jsonl(root / path, [{**defaults, **q} for q in questions])


def write_sources(root: Path, venues: Iterable[str]) -> None:
    """data/sources.yaml naming `venues`, each detected by its own name only."""
    entries = [
        {
            "venue": v,
            "venue_type": "cex",
            "kind": "gitbook",
            "license": "x",
            "llms_index": f"https://{v}.invalid/llms.txt",
        }
        for v in venues
    ]
    write_file(root, "data/sources.yaml", yaml.safe_dump({"sources": entries}))


def write_judged_run(
    root: Path,
    run_id: str = JUDGED_V1,
    *,
    records: Iterable[dict[str, Any]] = (),
    paired: Iterable[dict[str, Any]] = (),
    config: dict[str, Any] | None = None,
) -> None:
    """A judged answer run under results/: answers.jsonl, paired.csv and config.yaml."""
    directory = root / "results" / run_id
    write_jsonl(directory / "answers.jsonl", list(records))
    write_file(root, f"results/{run_id}/config.yaml", yaml.safe_dump(config or ANSWER_CONFIG))
    if paired:
        write_csv(directory / "paired.csv", complete("answer_paired", paired))

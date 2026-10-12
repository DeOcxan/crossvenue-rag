"""Tests for `reporting.sources`: the loaders over a fixture run."""

from __future__ import annotations

import re

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.models import RETRIEVAL_STEPS
from crossvenue_rag.reporting.run_ids import JUDGED_V1
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import REPO
from tests.generation.factories import generation_json, judge_json, judgement_json, record_json
from tests.reporting.conftest import write_fixture, write_judged_run


def test_the_answer_runs_load_their_records(tmp_path):
    write_fixture(tmp_path)
    src = Sources(tmp_path)
    assert len(src.v1.records) == len(src.v2.records) == 8
    assert src.questions["cv-0002"].expected_behavior == "refuse"


def test_a_claim_citing_a_chunk_outside_its_context_is_refused_where_answers_are_read(tmp_path):
    generation = generation_json(chunk_ids=("c1",))
    generation["answer"]["claims"][0]["chunks"] = [2]
    write_judged_run(tmp_path, records=[record_json(generation=generation)])
    with pytest.raises(UserError, match=r"cites chunk 2 of a context of 1 chunk$"):
        _ = Sources(tmp_path).judged_v1.records


def test_a_judgment_of_more_claims_than_the_answer_has_is_refused_where_answers_are_read(
    tmp_path,
):
    judge = judge_json(judgement=judgement_json(claims=2))
    write_judged_run(tmp_path, JUDGED_V1, records=[record_json(judge=judge)])
    with pytest.raises(UserError, match="grades 2 claims of an answer with 1"):
        _ = Sources(tmp_path).judged_v1.records


def test_the_latency_table_reads_the_committed_and_the_formatted_report(tmp_path):
    """A committed run is read from its report.md as written; a run whose
    summary.csv predates the latency columns but whose report.md uses
    `formatting.ms` reads back as plain decimals."""

    from crossvenue_rag.reporting.sources import RetrievalRun

    committed = REPO / "results/20260925T012314Z"
    table = RetrievalRun(committed).step_table
    assert table["hybrid_rerank"]["rerank"] == "2505.3"
    assert table["bm25"]["search"] == "0.1"

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.csv").write_text(
        "method,questions,labelled_questions,p50_ms,p95_ms\n"
        "bm25,1,1,0.1,0.2\nhybrid_rerank,1,1,2531.3,2600.0\n"
    )
    (run / "report.md").write_text(
        "# Retrieval ablation\n\n## Latency (p50, ms per step)\n\n"
        "| method | encode | search | fuse | rerank | steps sum p50 |\n"
        "|---|---|---|---|---|---|\n"
        "| bm25 | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 |\n"
        "| hybrid_rerank | 16 | 12 | 0.1 | 2,505 | 2,531 |\n"
        "\nA later table of the section is not read.\n\n| other | x |\n|---|---|\n| y | z |\n",
        encoding="utf-8",
    )
    formatted = RetrievalRun(run).step_table
    assert set(formatted) == {"bm25", "hybrid_rerank"}
    assert formatted["hybrid_rerank"]["rerank"] == "2505"
    assert set(formatted["hybrid_rerank"]) == set(RETRIEVAL_STEPS)
    assert formatted["bm25"]["search"] == "0.1"


@pytest.mark.parametrize(
    ("report", "problem"),
    [
        (None, "cannot be read"),
        ("# Retrieval ablation\n\nNo latency here.\n", "has no table under"),
        ("## Latency (p50, ms per step)\n\nNo table.\n", "has no table under"),
        (
            "## Latency (p50, ms per step)\n\n| method | encode | search | fuse | rerank |\n"
            "|---|---|---|---|---|\n| bm25 |\n",
            "has 1 cell, its header 5",
        ),
    ],
)
def test_a_report_latency_table_that_cannot_be_read_is_a_user_error(tmp_path, report, problem):
    from crossvenue_rag.reporting.sources import RetrievalRun

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.csv").write_text("method,questions,labelled_questions,p50_ms,p95_ms\n")
    if report is not None:
        (run / "report.md").write_text(report, encoding="utf-8")
    with pytest.raises(UserError, match=problem):
        _ = RetrievalRun(run).step_table


def test_a_report_latency_table_without_a_step_column_is_a_user_error(tmp_path):
    from crossvenue_rag.reporting.sources import RetrievalRun

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.csv").write_text("method,questions,labelled_questions,p50_ms,p95_ms\n")
    (run / "report.md").write_text(
        "## Latency (p50, ms per step)\n\n"
        "| method | encode | search | total p50 |\n|---|---|---|---|\n| bm25 | 0.0 | 0.1 | 0.1 |\n",
        encoding="utf-8",
    )
    lacking = r"the latency table lacks 2 step columns: fuse and rerank$"
    with pytest.raises(UserError, match=lacking):
        _ = RetrievalRun(run).step_table


def test_a_summary_with_some_step_columns_but_not_all_is_a_user_error(tmp_path):
    from crossvenue_rag.reporting.sources import RetrievalRun

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.csv").write_text(
        "method,questions,labelled_questions,p50_ms,p95_ms,encode_p50_ms,search_p50_ms\n"
        "bm25,1,1,0.1,0.2,0.0,0.1\n"
    )
    lacking = r"summary\.csv lacks 2 step columns: fuse_p50_ms and rerank_p50_ms$"
    with pytest.raises(UserError, match=lacking):
        _ = RetrievalRun(run).step_table


def test_the_latency_table_comes_from_summary_columns_when_the_run_has_them(tmp_path):
    from crossvenue_rag.reporting.sources import RetrievalRun

    run = tmp_path / "run"
    run.mkdir()
    steps = ["encode", "search", "fuse", "rerank", "total"]
    counts = ["questions", "labelled_questions"]
    header = ["method", *counts, "p50_ms", "p95_ms"]
    header += [f"{s}_{q}_ms" for s in steps for q in ("p50", "p95")]
    row = ["hybrid_rerank", "1", "1", "2531.3", "2600.0"]
    row += [str(1000.0 + i + 0.123) for i in range(10)]
    (run / "summary.csv").write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    table = RetrievalRun(run).step_table  # no report.md: never read
    assert table["hybrid_rerank"]["encode"] == "1000.123"
    assert table["hybrid_rerank"]["rerank"] == "1006.123"
    assert set(table["hybrid_rerank"]) == set(RETRIEVAL_STEPS)


@pytest.mark.parametrize("cell", ["", "abc", "nan"])
def test_a_summary_step_cell_that_is_no_number_is_a_user_error(tmp_path, cell):
    from crossvenue_rag.reporting.sources import RetrievalRun

    run = tmp_path / "run"
    run.mkdir()
    columns = [f"{s}_p50_ms" for s in RETRIEVAL_STEPS]
    (run / "summary.csv").write_text(
        "method,questions,labelled_questions,p50_ms,p95_ms," + ",".join(columns) + "\n"
        f"bm25,1,1,0.1,0.2,0.0,{cell},0.0,0.0\n"
    )
    with pytest.raises(
        UserError, match=f"summary\\.csv: the search p50 of bm25 is no number: '{cell}'$"
    ) as raised:
        _ = RetrievalRun(run).step_table
    assert raised.value.hint == "run `crossvenue eval` again, which writes every step column"


def test_a_report_step_cell_that_is_no_number_is_a_user_error(tmp_path):
    from crossvenue_rag.reporting.sources import RetrievalRun

    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.csv").write_text("method,questions,labelled_questions,p50_ms,p95_ms\n")
    (run / "report.md").write_text(
        "## Latency (p50, ms per step)\n\n"
        "| method | encode | search | fuse | rerank | total p50 |\n"
        "|---|---|---|---|---|---|\n| bm25 | 0.0 | x | 0.0 | 0.0 | 0.1 |\n",
        encoding="utf-8",
    )
    with pytest.raises(UserError, match=r"report\.md: the search p50 of bm25 is no number: 'x'$"):
        _ = RetrievalRun(run).step_table


# Keys of a run's saved config that `reporting` reads other than as
# `src.<run>.config.<section>.<key>`, which the test below finds in the code itself.
# A key its run did not save would read as today's default (`run_files.read_run_config`).
INDIRECT_CONFIG_KEYS = {
    "judged_v1": ["generation.judge", "generation.judges"],
}
# A method of a config section (`generation.chosen_judge()`) is not a key.
CONFIG_READ = re.compile(r"src\.(\w+)\.config\.(\w+)\.(\w+)\b(?!\()")


def config_reads() -> dict[str, set[str]]:
    """Every `src.<run>.config.<section>.<key>` in reporting and the figures script, by run."""
    code = [*(REPO / "src/crossvenue_rag/reporting").rglob("*.py"), REPO / "scripts/figures.py"]
    reads: dict[str, set[str]] = {run: set(keys) for run, keys in INDIRECT_CONFIG_KEYS.items()}
    for path in code:
        for run, section, key in CONFIG_READ.findall(path.read_text(encoding="utf-8")):
            reads.setdefault(run, set()).add(f"{section}.{key}")
    return reads


def test_every_config_key_reporting_reads_is_saved_by_its_run():
    src = Sources(REPO)
    reads = config_reads()
    assert "retrieval" in reads and "v1" in reads
    for run, keys in reads.items():
        config = getattr(src, run).config
        for key in sorted(keys):
            section, name = key.split(".")
            assert section in config.model_fields_set, (run, key)
            assert name in getattr(config, section).model_fields_set, (run, key)


def test_a_run_is_loaded_once_per_run_id_so_its_files_are_read_once(tmp_path):
    from crossvenue_rag.reporting.run_ids import RETRIEVAL_RUN

    src = Sources(tmp_path)
    assert src.run("20260101T000000Z") is src.run("20260101T000000Z")
    assert src.retrieval is src.run(RETRIEVAL_RUN)

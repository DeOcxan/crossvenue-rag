"""Tests for `crossvenue_rag.reporting.compute.bench`: pairing, counts and checks of a bench run."""

from __future__ import annotations

import pytest

from crossvenue_rag.evaluation.bench import BenchRow, BenchSummaryRow
from crossvenue_rag.evaluation.machine import (
    MACHINE_FILE,
    LoadDuring,
    MachineRecord,
    RecordedState,
    machine_json,
)
from crossvenue_rag.reporting.compute import bench, datasets
from crossvenue_rag.reporting.run_ids import BENCH_RUN, RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources
from crossvenue_rag.run_files import CsvTable, write_table
from tests.reporting.conftest import write_gold, write_retrieval_run, write_sources

# The bench's timings: (question, mode, position, total ms, rerank ms).
TIMINGS = (
    ("cv-0001", "per_venue", 1, 130.0, 100.0),
    ("cv-0001", "one_call", 2, 110.0, 90.0),
    ("cv-0002", "one_call", 1, 125.0, 95.0),
    ("cv-0002", "per_venue", 2, 125.0, 105.0),
)


def bench_row(question_id, mode, position, total_ms, rerank_ms):
    return BenchRow(
        question_id=question_id,
        round=1,
        position=position,
        mode=mode,
        total_ms=total_ms,
        rerank_ms=rerank_ms,
        calls=2 if mode == "per_venue" else 1,
        pairs_per_call="20 20" if mode == "per_venue" else "40",
        identical=True,
        load1=1.5,
    )


def summary_row(mode, *, pairs=40.0, identical=2, total_p50_ms=100.0):
    return BenchSummaryRow(
        mode=mode,
        n=2,
        total_p50_ms=total_p50_ms,
        total_p95_ms=130.0,
        total_mean_ms=122.5,
        rerank_p50_ms=95.0,
        rerank_p95_ms=105.0,
        rerank_mean_ms=97.5,
        calls_per_query=2.0 if mode == "per_venue" else 1.0,
        pairs_per_call_mean=20.0 if mode == "per_venue" else 40.0,
        pairs_per_query_mean=pairs,
        identical_retrievals=identical,
    )


def recorded(cpu_count):
    return RecordedState(
        measured_at="2026-01-01T00:00:00+00:00",
        cpu_count=cpu_count,
        load_average=(1.0, 1.0, 1.0),
        competing_processes=0,
        busy_processes=0,
    )


def write_bench(root, *, summary=None, cpu_count=8, load=None):
    directory = root / "results" / BENCH_RUN
    directory.mkdir(parents=True)
    write_table(directory / "raw.csv", CsvTable.of([bench_row(*t) for t in TIMINGS]))
    rows = summary or [summary_row("per_venue"), summary_row("one_call")]
    write_table(directory / "summary.csv", CsvTable.of(rows))
    record = MachineRecord(
        before=recorded(cpu_count),
        after=recorded(cpu_count),
        discarded_pairs=3,
        load1_during=load if load is not None else LoadDuring(median=1.5, max=2.5),
    )
    (directory / MACHINE_FILE).write_text(machine_json(record), encoding="utf-8")


def write_fixture(root, **bench_options):
    """Two questions venue_split splits and one it does not, the bench and the retrieval run."""
    write_sources(root, ["binance", "bybit", "okx"])
    write_gold(
        root,
        [
            {"id": "cv-0001", "question": "binance or bybit fees"},
            {"id": "cv-0002", "question": "okx and binance margin"},
            {"id": "cv-0003", "question": "binance deposit limits"},
        ],
    )
    per_query = [
        {"question_id": q, "method": "venue_split", "total_ms": ms}
        for q, ms in (("cv-0001", 80.0), ("cv-0002", 80.0), ("cv-0003", 10.0))
    ]
    config = {"eval": {"bootstrap_resamples": 200, "bootstrap_seed": 1}}
    write_retrieval_run(root, RETRIEVAL_RUN, config=config, per_query=per_query)
    write_bench(root, **bench_options)
    return Sources(root)


def test_paired_bench_pairs_the_modes_by_question():
    paired = bench.paired_bench([bench_row(*t) for t in TIMINGS], resamples=200, seed=1)
    assert paired.total_ms.value == pytest.approx(10.0)
    assert paired.rerank_ms.value == pytest.approx(10.0)
    assert paired.rerank_ms.low == pytest.approx(10.0)
    assert paired.rerank_ms.high == pytest.approx(10.0)


def test_one_call_faster_counts_the_questions_whose_rerank_step_one_call_won():
    rows = [bench_row(*t) for t in TIMINGS]
    rows.append(bench_row("cv-0003", "per_venue", 1, 100.0, 50.0))
    rows.append(bench_row("cv-0003", "one_call", 2, 120.0, 70.0))
    paired = bench.paired_bench(rows, resamples=200, seed=1)
    assert paired.one_call_faster == 2


def test_time_outside_the_rerank_step_is_averaged_by_position_in_the_pair():
    paired = bench.paired_bench([bench_row(*t) for t in TIMINGS], resamples=200, seed=1)
    # Timed first: 30 ms outside the rerank step on each question; timed second: 20 ms.
    assert paired.outside_first == pytest.approx(30.0)
    assert paired.outside_second == pytest.approx(20.0)


def test_the_bench_times_the_questions_venue_split_splits(tmp_path):
    src = write_fixture(tmp_path)
    assert bench.timed_question_count(src) == 2
    assert bench.times_split_questions(src)


def test_a_slice_below_the_tested_size_with_a_saving_excluding_zero_is_descriptive(tmp_path):
    src = write_fixture(tmp_path)
    rerank = bench.paired_difference(src, "rerank_ms")
    assert (rerank.value, rerank.low, rerank.high) == pytest.approx((10.0, 10.0, 10.0))
    assert bench.descriptive_with_saving(src)


def test_the_modes_agree_only_on_equal_pairs_and_identical_retrievals(tmp_path):
    assert bench.modes_agree(write_fixture(tmp_path / "same"))
    fewer_pairs = [summary_row("per_venue"), summary_row("one_call", pairs=39.0)]
    assert not bench.modes_agree(write_fixture(tmp_path / "pairs", summary=fewer_pairs))
    one_differs = [summary_row("per_venue"), summary_row("one_call", identical=1)]
    assert not bench.modes_agree(write_fixture(tmp_path / "ids", summary=one_differs))


def test_mode_rows_list_the_bench_columns_per_mode_in_order(tmp_path):
    rows = bench.mode_rows(write_fixture(tmp_path))
    assert [m for m, _ in rows] == ["per_venue", "one_call"]
    assert rows[1][1] == [1.0, 40.0, 100.0, 130.0, 122.5]


def test_a_machine_record_without_load_or_cpus_is_an_error(tmp_path):
    src = write_fixture(tmp_path / "full")
    assert bench.load_average(src) == (1.5, 2.5)
    assert bench.cpu_count(src) == 8
    bare = write_fixture(tmp_path / "bare", cpu_count=None, load=LoadDuring())
    with pytest.raises(ValueError, match="records no load while timing"):
        bench.load_average(bare)
    with pytest.raises(ValueError, match="records no CPU count"):
        bench.cpu_count(bare)


def test_evaluation_p50_below_the_bench_is_given_with_the_bench_p50(tmp_path):
    rows = [summary_row("per_venue", total_p50_ms=100.0), summary_row("one_call")]
    src = write_fixture(tmp_path, summary=rows)
    # venue_split's p50 on the two split questions is 80 ms; cv-0003 is not split.
    assert bench.eval_below_bench(src) == pytest.approx((20.0, 100.0))


def test_timing_as_many_questions_as_venue_split_splits_but_others_is_refused(tmp_path):
    write_fixture(tmp_path)
    write_gold(
        tmp_path,
        [
            {"id": "cv-0001", "question": "binance or bybit fees"},
            {"id": "cv-0002", "question": "binance margin"},
            {"id": "cv-0003", "question": "okx and binance deposit limits"},
        ],
    )
    src = Sources(tmp_path)
    assert bench.timed_question_count(src) == len(datasets.split_ids(src)) == 2
    assert not bench.times_split_questions(src)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([bench_row(*t) for t in TIMINGS[:3]], "raw.csv of cv-0002 has no 'per_venue'"),
        ([bench_row("cv-0001", m, 3, 1.0, 1.0) for m in ("per_venue", "one_call")], "has no 3"),
    ],
)
def test_a_question_without_both_modes_or_a_third_position_is_a_value_error(rows, message):
    with pytest.raises(ValueError, match=message):
        bench.paired_bench(rows, resamples=200, seed=1)


def test_a_summary_without_a_mode_is_a_value_error(tmp_path):
    src = write_fixture(tmp_path, summary=[summary_row("one_call")])
    with pytest.raises(ValueError, match=r"summary\.csv has no 'per_venue'"):
        bench.mode_value(src, "per_venue", "total_p50_ms")
    with pytest.raises(ValueError, match=r"summary\.csv has no 'per_venue'"):
        bench.modes_agree(src)

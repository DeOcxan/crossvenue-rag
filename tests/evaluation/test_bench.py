"""Tests for `evaluation.bench`: alternation, the summary, the busy check, the files."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import ClassVar

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.errors import UserError
from crossvenue_rag.evaluation.bench import (
    BENCH_REPORT_FILE,
    MAX_PAIR_RETRIES,
    MAX_QUIET_CHECKS,
    BenchRequest,
    BenchResult,
    MachineProbes,
    Pacing,
    QuietCheck,
    bench_rerank,
    read_bench_raw,
    read_bench_summary,
    run_bench,
    split_questions,
    summarise_modes,
    write_bench,
)
from crossvenue_rag.evaluation.machine import (
    MACHINE_FILE,
    MachineState,
    Process,
)
from crossvenue_rag.models import Question, Result
from crossvenue_rag.run_files import write_table
from tests.conftest import committed_with, default_config, output_defects


class Detector:
    def detect(self, text):
        return [v for v in ("binance", "bybit") if v in text]


class TimedSplitRetriever:
    """Per-venue calls take 3 ms of rerank each, one call 5 ms; 50 pairs per venue."""

    venue_detector = Detector()

    def __init__(self, differ_on: str = ""):
        self.calls: list[tuple[str, str]] = []
        self.differ_on = differ_on

    def venue_split_timed(self, query, rerank_calls, timings, pairs_per_call):
        self.calls.append((query, rerank_calls))
        venues = self.venue_detector.detect(query)
        if rerank_calls == "one_call":
            pairs_per_call.append(50 * len(venues))
            timings.rerank_ms += 5.0
        else:
            pairs_per_call.extend([50] * len(venues))
            timings.rerank_ms += 3.0 * len(venues)
        timings.encode_ms += 1.0
        ids = ["a", "b"] if not (query == self.differ_on and rerank_calls == "per_venue") else ["b"]
        return [Result(i, 1.0, "t", "binance", "cex", "d", [], "u", sources=[]) for i in ids]


# No quiet-machine checks, no wait, a load of 0.0 on every row.
UNPACED = Pacing(quiet=None, wait=lambda: None, load=lambda: 0.0)


def questions():
    return [
        Question("cv-1", "binance vs bybit limits", "gold", ["binance", "bybit"], "comparison"),
        Question("cv-2", "binance only", "gold", ["binance"], "fact"),
        Question("cv-3", "bybit or binance fees", "gold", ["binance", "bybit"], "comparison"),
    ]


def test_only_questions_venue_split_splits_are_timed():
    assert [q.id for q in split_questions(TimedSplitRetriever(), questions())] == ["cv-1", "cv-3"]


def test_modes_alternate_in_order_between_questions_and_rounds():
    retriever = TimedSplitRetriever()
    chosen = split_questions(retriever, questions())
    rows, discarded = run_bench(retriever, chosen, rounds=2, pacing=UNPACED)
    assert discarded == 0
    modes = [mode for _, mode in retriever.calls]
    assert modes == [
        "per_venue", "one_call",  # cv-1, round 1
        "one_call", "per_venue",  # cv-3, round 1
        "one_call", "per_venue",  # cv-1, round 2
        "per_venue", "one_call",  # cv-3, round 2
    ]  # fmt: skip
    assert len(rows) == 8
    first = [r for r in rows if r.position == 1]
    assert sorted(r.mode for r in first) == ["one_call", "one_call", "per_venue", "per_venue"]


def test_the_summary_counts_calls_and_pairs_and_flags_different_retrievals():
    retriever = TimedSplitRetriever(differ_on="bybit or binance fees")
    rows, _ = run_bench(retriever, split_questions(retriever, questions()), 1, UNPACED)
    summary = {s.mode: s for s in summarise_modes(rows)}
    assert summary["per_venue"].calls_per_query == 2.0
    assert summary["one_call"].calls_per_query == 1.0
    assert summary["per_venue"].pairs_per_call_mean == 50.0
    assert summary["one_call"].pairs_per_call_mean == 100.0
    assert summary["per_venue"].pairs_per_query_mean == summary["one_call"].pairs_per_query_mean
    assert summary["per_venue"].rerank_p50_ms == 6.0
    assert summary["one_call"].rerank_p50_ms == 5.0
    assert summary["one_call"].identical_retrievals == 1


def test_the_run_directory_holds_raw_times_summary_and_machine_state(tmp_path):
    retriever = TimedSplitRetriever()
    rows, _ = run_bench(retriever, split_questions(retriever, questions()), 1, UNPACED)
    competitor = Process(pid=7, cpu_percent=90.0, command="other-project run")
    state = MachineState(
        measured_at="t",
        cpu_count=8,
        load_average=(0.5, 0.5, 0.5),
        competing_processes=(competitor,),
        busy_processes=(),
    )
    questions_file = tmp_path / "q.jsonl"
    questions_file.write_text('{"id": "cv-1"}\n')
    result = BenchResult(rows, summarise_modes(rows), state, state, discarded=0, checked=True)
    run_dir = write_bench(default_config(tmp_path), result, "abc", questions_file)
    index = json.loads((run_dir / "index.json").read_text())
    assert index["questions_sha256"] == hashlib.sha256(questions_file.read_bytes()).hexdigest()
    raw = list(csv.DictReader((run_dir / "raw.csv").open()))
    assert len(raw) == 4 and raw[0]["pairs_per_call"] == "50 50"
    assert [r["mode"] for r in csv.DictReader((run_dir / "summary.csv").open())] == [
        "per_venue",
        "one_call",
    ]
    machine = json.loads((run_dir / "machine.json").read_text())
    assert machine["before"]["cpu_count"] == 8
    # Counts only: another process's command line never reaches the saved run.
    assert machine["before"]["competing_processes"] == 1
    assert "other-project" not in (run_dir / "machine.json").read_text()
    bench_md = (run_dir / BENCH_REPORT_FILE).read_text()
    assert "identical retrievals" in bench_md
    assert output_defects(bench_md) == []
    assert "0.50, 0.50 and 0.50 on 8 CPUs" in bench_md
    assert (
        "none of the competing processes (crossvenue, pytest, uvicorn and sentence_transformers)"
        in bench_md
    )


def test_a_forced_bench_says_its_quiet_machine_checks_were_skipped(tmp_path):
    retriever = TimedSplitRetriever()
    rows, _ = run_bench(retriever, split_questions(retriever, questions()), 1, UNPACED)
    state = IDLE
    questions_file = tmp_path / "q.jsonl"
    questions_file.write_text('{"id": "cv-1"}\n')
    result = BenchResult(rows, summarise_modes(rows), state, state, discarded=0, checked=False)
    run_dir = write_bench(default_config(tmp_path), result, "abc", questions_file)
    bench_md = (run_dir / BENCH_REPORT_FILE).read_text()
    assert "The quiet-machine checks were skipped (`--force`)" in bench_md
    assert (
        "none of the competing processes (crossvenue, pytest, uvicorn and sentence_transformers)"
        not in bench_md
    )
    assert output_defects(bench_md) == []


def test_a_pair_with_a_competitor_at_its_end_is_discarded_and_repeated():
    retriever = TimedSplitRetriever()
    # quiet() returns: before q1, after q1 (busy: discard), busy, quiet before
    # the retry, after the retry, before q3, after q3.
    answers = iter([True, False, False, True, True, True, True])
    waits = []
    rows, discarded = run_bench(
        retriever,
        split_questions(retriever, questions()),
        rounds=1,
        pacing=Pacing(quiet=lambda: next(answers), wait=lambda: waits.append(1), load=lambda: 1.5),
    )
    assert discarded == 1 and waits == [1]
    assert [r.question_id for r in rows] == ["cv-1", "cv-1", "cv-3", "cv-3"]
    assert len(retriever.calls) == 6  # q1 twice, q3 once
    assert all(r.load1 == 1.5 for r in rows)


def test_a_machine_that_never_goes_quiet_is_a_user_error_naming_force():
    retriever = TimedSplitRetriever()
    busy_after = itertools.cycle([True, False])
    with pytest.raises(
        UserError, match=f"busy after each of {MAX_PAIR_RETRIES} timed pairs"
    ) as info:
        run_bench(
            retriever,
            split_questions(retriever, questions()),
            rounds=1,
            pacing=Pacing(quiet=busy_after.__next__, wait=lambda: None, load=lambda: 0.0),
        )
    assert "--force" in (info.value.hint or "")


def test_a_machine_busy_before_a_pair_gives_up_after_the_named_checks_naming_force():
    retriever = TimedSplitRetriever()
    waits = []
    with pytest.raises(UserError, match=f"busy at each of {MAX_QUIET_CHECKS} checks") as info:
        run_bench(
            retriever,
            split_questions(retriever, questions()),
            rounds=1,
            pacing=Pacing(quiet=lambda: False, wait=lambda: waits.append(1), load=lambda: 0.0),
        )
    assert "--force" in (info.value.hint or "")
    assert len(waits) == MAX_QUIET_CHECKS and retriever.calls == []


IDLE = MachineState(
    measured_at="t",
    cpu_count=8,
    load_average=(0.5, 0.5, 0.5),
    competing_processes=(),
    busy_processes=(),
)
BUSY = IDLE.model_copy(
    update={"competing_processes": (Process(pid=7, cpu_percent=90.0, command="pytest"),)}
)


class OpenedRetriever(TimedSplitRetriever):
    """TimedSplitRetriever as `bench_rerank` opens it: a context manager with an index."""

    opened: ClassVar[list[OpenedRetriever]] = []

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.warmed = False
        OpenedRetriever.opened.append(self)

    def require_current_index(self) -> str:
        return "idx-bench"

    def warmup(self, *, load_reranker: bool = True) -> None:
        self.warmed = True

    def __enter__(self) -> OpenedRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def states(*answers: MachineState) -> tuple[MachineProbes, list[int]]:
    """Probes whose state replays `answers` (the last one repeats), and their call log.

    Their load is 0.25 and their sleep returns at once.
    """
    calls: list[int] = []

    def fake() -> MachineState:
        calls.append(1)
        return answers[min(len(calls), len(answers)) - 1]

    return MachineProbes(fake, lambda: 0.25, lambda _seconds: None), calls


def bench_questions(tmp_path: Path) -> tuple[Path, list[Question]]:
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "cv-1"}\n', encoding="utf-8")
    return path, questions()


def test_bench_rerank_warms_up_times_the_split_questions_and_writes_the_run(tmp_path):
    probes, _ = states(IDLE)
    cfg = default_config(tmp_path)
    path, qs = bench_questions(tmp_path)
    request = BenchRequest(path, rounds=2, force=False)
    run_dir = bench_rerank(
        cfg, request, qs, notify=lambda _msg: None, probes=probes, open_retriever=OpenedRetriever
    )
    assert run_dir.parent == tmp_path / "results" and run_dir.name.endswith("_bench")
    assert cfg.eval.questions != path  # the caller's config is not changed
    assert f"questions: {path}" in (run_dir / "config.yaml").read_text(encoding="utf-8")
    retriever = OpenedRetriever.opened[-1]
    assert retriever.warmed
    # One warm-up call per mode on the first split question, then 2 questions x 2 rounds x 2.
    assert len(retriever.calls) == 2 + 8
    raw = list(csv.DictReader((run_dir / "raw.csv").open()))
    assert {r["question_id"] for r in raw} == {"cv-1", "cv-3"} and len(raw) == 8
    assert {r["load1"] for r in raw} == {"0.25"}  # the load the probes read
    assert json.loads((run_dir / "index.json").read_text())["index_version"] == "idx-bench"


def test_bench_rerank_with_force_never_checks_for_a_quiet_machine(tmp_path):
    probes, calls = states(BUSY)
    cfg = default_config(tmp_path)
    path, qs = bench_questions(tmp_path)
    request = BenchRequest(path, rounds=1, force=True)
    run_dir = bench_rerank(
        cfg, request, qs, notify=lambda _msg: None, probes=probes, open_retriever=OpenedRetriever
    )
    assert len(calls) == 2  # the states before and after, nothing in between
    assert json.loads((run_dir / "machine.json").read_text())["before"]["competing_processes"] == 1


def test_bench_rerank_refuses_questions_that_name_one_venue(tmp_path):
    probes, _ = states(IDLE)
    path, qs = bench_questions(tmp_path)
    request = BenchRequest(path, rounds=1, force=True)
    cfg = default_config(tmp_path)
    with pytest.raises(UserError, match="no question names 2 or more venues"):
        bench_rerank(
            cfg,
            request,
            [qs[1]],
            notify=lambda _msg: None,
            probes=probes,
            open_retriever=OpenedRetriever,
        )


def test_the_quiet_check_announces_each_wait_once():
    probes, _ = states(BUSY, BUSY, IDLE, BUSY)
    heard: list[str] = []
    check = QuietCheck(heard.append, probes.state)
    assert [check(), check(), check(), check()] == [False, False, True, False]
    # Busy twice in a row is one wait; quiet in between starts another.
    assert heard == ["pid 7: pytest", "pid 7: pytest"]


def test_a_bench_without_quiet_checks_times_each_pair_once_without_waiting():
    retriever = TimedSplitRetriever()
    chosen = split_questions(retriever, questions())
    rows, discarded = run_bench(retriever, chosen, rounds=1, pacing=UNPACED)
    assert discarded == 0
    assert len(retriever.calls) == len(rows) == 2 * len(chosen)


@pytest.mark.parametrize("run", committed_with(MACHINE_FILE), ids=lambda p: p.name)
def test_every_committed_bench_csv_round_trips_byte_for_byte(run, tmp_path):
    for name, read in (("raw.csv", read_bench_raw), ("summary.csv", read_bench_summary)):
        write_table(tmp_path / name, read(run))
        assert (tmp_path / name).read_bytes() == (run / name).read_bytes(), name


def test_the_report_verb_agrees_with_the_count_of_discarded_pairs():
    from crossvenue_rag.evaluation.bench import bench_report

    retriever = TimedSplitRetriever()
    rows, _ = run_bench(retriever, split_questions(retriever, questions()), 1, UNPACED)
    summary = summarise_modes(rows)
    one = bench_report(rows, summary, IDLE, discarded=1, checked=True)
    # Both reasons a second check fails are named: a competing process and the load.
    assert "running and a 1-minute load average within the bound; 1 pair that failed" in one
    assert "the second check, on either count, was discarded and repeated." in one
    two = bench_report(rows, summary, IDLE, discarded=2, checked=True)
    assert "2 pairs that failed the second check, on either count, were discarded" in two


def test_force_is_passed_by_keyword_only(tmp_path):
    with pytest.raises(TypeError):
        # The positional boolean is the call under test.
        BenchRequest(tmp_path / "q.jsonl", 2, True)  # type: ignore[call-arg]  # noqa: FBT003
    assert BenchRequest(tmp_path / "q.jsonl", 2, force=True).force

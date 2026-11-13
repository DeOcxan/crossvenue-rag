"""Tests for `cli.labeling`: the labeling commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.cli import labeling as _under_test
from crossvenue_rag.errors import UserError
from tests.cli.conftest import current_version, invoke
from tests.conftest import write_journal


def _labelled_questions(tmp_path: Path, version: str) -> Path:
    """One comparison question with a round-1 judgment against index `version`."""
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.models import Question

    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit", "okx"], "comparison")])
    write_journal(
        path, {"q1": {"1": {"pool": ["c1", "c2"], "relevant": ["c1"], "index_version": version}}}
    )
    return path


def test_label_passes_run_label_its_session_options(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.cli import labeling as modes

    got: list = []
    monkeypatch.setattr(modes, "run_label", lambda cfg, request: got.append((cfg, request)))
    sample = tmp_path / "sample.txt"
    result = invoke(
        "label", "cv-0001", "cv-0002", "-c", str(config_file), "--review",
        "--all", "--limit", "7", "--pool-depth", "30", "--round", "2",
        "--annotator", "me", "--sample", "3", "--seed", "4", "--sample-file", str(sample),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    [(cfg, request)] = got
    assert request.questions_path == str(cfg.eval.questions)
    assert request.ids == ("cv-0001", "cv-0002") and request.review and request.revisit
    assert (request.limit, request.pool_depth, request.annotation_round) == (7, 30, 2)
    assert (request.annotator, request.sample, request.seed) == ("me", 3, 4)
    assert request.sample_file == sample and not request.augments


@pytest.mark.parametrize("option", ["--sample", "--limit"])
def test_label_refuses_a_count_below_one(config_file, option):
    result = invoke("label", option, "0", "-c", str(config_file))
    assert isinstance(result.exception, UserError)
    assert f"{option} must be at least 1" in str(result.exception)


@pytest.mark.parametrize("number", ["0", "4"])
def test_label_refuses_a_round_outside_the_protocol(config_file, number):
    result = invoke("label", "--round", number, "-c", str(config_file))
    assert isinstance(result.exception, UserError)
    assert f"there is no round {number}" in str(result.exception)
    assert "--round 1 to 3" in result.exception.hint


@pytest.mark.parametrize("depth", ["0", "10"])
def test_label_refuses_a_pool_depth_within_the_scored_ranks(config_file, depth):
    result = invoke("label", "--pool-depth", depth, "-c", str(config_file))
    assert isinstance(result.exception, UserError)
    assert f"--pool-depth {depth}: a pool depth of {depth} does not exceed" in str(result.exception)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["-m", "nonsense"], "nonsense"),
        (["-m", "hybrid", "--category", "nonsense"], "is not a question category"),
        (["-m", "hybrid", "--k", "10"], "--k 10: a pool depth of 10 does not exceed"),
    ],
)
def test_augment_candidates_refuses_a_bad_option_and_writes_nothing(
    config_file, tmp_path, options, message
):
    out = tmp_path / "c.jsonl"
    result = invoke(
        "augment-candidates", "--run", str(tmp_path), "--out", str(out), "-c", str(config_file),
        *options,
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert message in str(result.exception)
    assert not out.exists()


def test_label_augment_options_reach_the_request(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.cli import labeling as modes

    got: list = []
    monkeypatch.setattr(modes, "run_label", lambda cfg, request: got.append(request))
    listing = tmp_path / "c.jsonl"
    questions = tmp_path / "q.jsonl"
    result = invoke(
        "label", "-c", str(config_file), "-q", str(questions),
        "--augment", "--question-file", str(listing), "--review-file", str(tmp_path / "r.txt"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    [request] = got
    assert request.augment and request.question_file == listing
    assert request.questions_path == str(questions) and request.ids == ()
    assert request.review_file == tmp_path / "r.txt"


def test_label_adjudicate_and_relabel_options_reach_the_request(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.cli import labeling as modes

    got: list = []
    monkeypatch.setattr(modes, "run_label", lambda cfg, request: got.append(request))
    added = tmp_path / "added.jsonl"
    assert invoke("label", "-c", str(config_file), "--adjudicate").exit_code == 0
    result = invoke(
        "label", "cv-0106", "-c", str(config_file), "--relabel", "--add-chunks", str(added)
    )
    assert result.exit_code == 0, result.output
    adjudicate, relabel = got
    assert adjudicate.adjudicate and not adjudicate.relabel and adjudicate.ids == ()
    assert relabel.relabel and relabel.ids == ("cv-0106",) and relabel.add_chunks == added


def test_label_refuses_ids_without_review_before_opening_the_index(config_file):
    result = invoke("label", "cv-0001", "-c", str(config_file))
    assert isinstance(result.exception, UserError)
    assert "without --review" in str(result.exception)


def _write_run(run: Path, version: str) -> None:
    run.mkdir()
    (run / "index.json").write_text(json.dumps({"index_version": version}), encoding="utf-8")
    (run / "per_query.csv").write_text(
        "question_id,method,retrieved_ids\nq1,hybrid,c1 n1 n2\nq1,bm25,n3 c2\nq2,hybrid,n9\n",
        encoding="utf-8",
    )


def test_augment_candidates_lists_the_chunks_round_1_has_not_judged(config_file, tmp_path):
    from crossvenue_rag.labeling.candidates import load_candidates

    questions = _labelled_questions(tmp_path, "v1")
    run, out = tmp_path / "run", tmp_path / "augment.jsonl"
    _write_run(run, "v1")
    result = invoke(
        "augment-candidates", "--run", str(run), "--out", str(out), "-m", "hybrid",
        "--k", "12", "-c", str(config_file), "-q", str(questions),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "2 chunks round 1 has not judged, across 1 question" in result.output
    assert "crossvenue label --augment --question-file" in result.output
    [entry] = load_candidates(out)
    assert [c.chunk_id for c in entry.chunks] == ["n1", "n2"] and entry.k == 12


def test_augment_candidates_refuses_to_overwrite_a_list(config_file, tmp_path):
    out = tmp_path / "augment.jsonl"
    out.write_text("", encoding="utf-8")
    result = invoke(
        "augment-candidates", "--run", str(tmp_path), "--out", str(out), "-m", "hybrid",
        "-c", str(config_file),
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert "exists" in str(result.exception)


@pytest.mark.parametrize(
    ("place", "names"),
    [
        ("absent/augment.jsonl", "the directory of --out, is not a directory"),
        ("file.txt/augment.jsonl", "file.txt is a file"),
    ],
)
def test_augment_candidates_refuses_an_out_path_in_a_missing_directory_or_under_a_file(
    config_file, tmp_path, place, names
):
    questions = _labelled_questions(tmp_path, "v1")
    run = tmp_path / "run"
    _write_run(run, "v1")
    (tmp_path / "file.txt").write_text("", encoding="utf-8")
    result = invoke(
        "augment-candidates", "--run", str(run), "--out", str(tmp_path / place), "-m", "hybrid",
        "--k", "12", "-c", str(config_file), "-q", str(questions),
    )  # fmt: skip
    assert isinstance(result.exception, UserError), result.exception
    assert names in str(result.exception)
    assert not (tmp_path / "absent").exists()


def test_derive_coverage_prints_each_update_and_the_exclusions(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.labeling import coverage

    questions = tmp_path / "q.jsonl"
    questions.write_text("", encoding="utf-8")
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("cv-0009\ncv-0010  reason\n", encoding="utf-8")
    got: list = []

    def derive(path, chunk_venues, version, excluded):
        got.append((path, version, set(excluded)))
        return [
            coverage.Derived("q1", {"bybit": "covered"}, []),
            coverage.Derived("q2", {"okx": "covered"}, ["bybit"]),
        ]

    monkeypatch.setattr(_under_test, "derive_coverage", derive)
    result = invoke(
        "derive-coverage", "-c", str(config_file), "-q", str(questions),
        "--exclude-file", str(exclude),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert got == [(str(questions), current_version(config_file), {"cv-0009", "cv-0010"})]
    assert "rule: covered" in result.output
    assert "q1: bybit covered" in result.output
    assert "undecided: bybit" in result.output
    assert "2 questions updated, 2 excluded" in result.output


def test_derive_coverage_with_nothing_to_derive_updates_and_excludes_none(config_file, tmp_path):
    questions = _labelled_questions(tmp_path, current_version(config_file))
    result = invoke("derive-coverage", "-c", str(config_file), "-q", str(questions))
    assert result.exit_code == 0, result.output
    assert "0 questions updated, 0 excluded" in result.output


def test_derive_coverage_refuses_labels_from_another_index(config_file, tmp_path):
    questions = _labelled_questions(tmp_path, "stale")
    result = invoke("derive-coverage", "-c", str(config_file), "-q", str(questions))
    assert isinstance(result.exception, UserError)
    assert "judged against stale" in str(result.exception)


def test_derive_coverage_without_chunks_names_the_file_and_the_command_that_writes_it(
    config_file, tmp_path
):
    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.corpus.artifacts import chunks_path

    chunks_path(load_config(config_file)).unlink()
    questions = _labelled_questions(tmp_path, current_version(config_file))
    result = invoke("derive-coverage", "-c", str(config_file), "-q", str(questions))
    assert isinstance(result.exception, UserError)
    assert "chunks.jsonl not found" in str(result.exception)
    assert "crossvenue ingest" in (result.exception.hint or "")


def test_label_round_3_options_reach_the_request(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.cli import labeling as modes

    got: list = []
    monkeypatch.setattr(modes, "run_label", lambda cfg, request: got.append(request))
    added, metadata = tmp_path / "added.jsonl", tmp_path / "base.jsonl"
    result = invoke(
        "label", "-c", str(config_file), "--round", "3",
        "--add-chunks", str(added), "--metadata-from", str(metadata),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert got[0].annotation_round == 3
    assert (got[0].add_chunks, got[0].metadata_from) == (added, metadata)

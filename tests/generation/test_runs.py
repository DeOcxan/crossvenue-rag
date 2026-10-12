"""Tests for `generation.runs`: the index, prompt and calibration gates on a run."""

from __future__ import annotations

import hashlib

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.calibration_file import calibration_json
from crossvenue_rag.generation.runs import (
    answer_run_index_version,
    generator_prompt,
    judge_prompt,
    load_prompt,
    require_run_index,
    uncalibrated_metrics,
)
from crossvenue_rag.run_files import RUNS_REGISTRY, write_index
from tests.conftest import calibration_of, registry_json_of
from tests.generation.offline import (
    gen_config,
)


def test_a_runs_index_version_is_read_from_its_index_record(tmp_path):
    write_index(tmp_path, {"index_version": "v1"}, final_newline=True)
    assert answer_run_index_version(tmp_path, tmp_path / "none.json") == "v1"


def test_a_saved_run_without_index_json_is_read_from_the_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "20260101T000000Z_answers"
    run.mkdir()
    (run / "answers.jsonl").write_text("", encoding="utf-8")
    registry = tmp_path / RUNS_REGISTRY
    assert answer_run_index_version(run, registry) is None
    registry.parent.mkdir()
    registry.write_text(registry_json_of({run.name: {"index_version": "v1"}}))
    assert answer_run_index_version(run, registry) == "v1"
    assert require_run_index(run, "v1") == "v1"
    with pytest.raises(UserError, match="made against index v1, the current index is v2"):
        require_run_index(run, "v2")


def test_a_run_with_no_version_anywhere_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "answers.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(UserError, match="records no index_version"):
        require_run_index(tmp_path, "v1")


def test_a_directory_that_is_not_an_answer_run_is_refused_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UserError, match=r"is not an answer run: it has no answers\.jsonl"):
        require_run_index(tmp_path / "missing", "v1")


def test_a_prompt_whose_file_changed_is_refused(tmp_path):
    path = tmp_path / "judge.md"
    path.write_text("<!-- judge prompt v2 -->\nGrade.\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_prompt(path, digest) == (path.read_text(encoding="utf-8"), digest)
    assert load_prompt(path, expected_sha256=None)[1] == digest
    with pytest.raises(UserError, match=f"has sha256 {digest}; the config pins {'0' * 64}$"):
        load_prompt(path, "0" * 64)
    cfg = gen_config(tmp_path, judge_prompt=path, judge_prompt_sha256=digest)
    assert judge_prompt(cfg)[1] == digest


def test_the_generator_prompt_version_must_match_the_answer_schema(tmp_path):
    path = tmp_path / "generator.md"
    path.write_text("<!-- generator prompt v1 (2026-09-25). -->\nAnswer.\n", encoding="utf-8")
    assert generator_prompt(gen_config(tmp_path, generator_prompt=path, answer_schema_version=1))
    with pytest.raises(
        UserError, match=r"is prompt v1, generation\.answer_schema_version is 2"
    ) as refused:
        generator_prompt(gen_config(tmp_path, generator_prompt=path, answer_schema_version=2))
    assert refused.value.hint.startswith(
        "set generation.answer_schema_version to 1, or set generation.generator_prompt "
        "to a prompt v2 file"
    )
    path.write_text("Answer.\n", encoding="utf-8")
    with pytest.raises(UserError, match="has no version line"):
        generator_prompt(gen_config(tmp_path, generator_prompt=path))


def test_metrics_of_a_failed_dimension_are_uncalibrated(tmp_path):
    (tmp_path / "calibration.json").write_text(
        calibration_json(
            calibration_of(candidates={"opus": {"passes": ["coverage"], "fails": ["supported"]}})
        )
    )
    cfg = gen_config(tmp_path, calibration_run=tmp_path)
    assert uncalibrated_metrics(cfg, "opus") == ["faithfulness"]
    assert uncalibrated_metrics(cfg, None) == []
    # A judge no calibration names is calibrated on nothing.
    assert len(uncalibrated_metrics(cfg, "sonnet")) == 4
    assert len(uncalibrated_metrics(gen_config(tmp_path), "opus")) == 4


def test_a_pinned_prompt_is_read_once_however_often_a_run_asks_for_it(tmp_path, monkeypatch):
    from pathlib import Path

    prompt = tmp_path / "judge.md"
    prompt.write_text("grade the claims", encoding="utf-8")
    digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
    reads: list[Path] = []
    real = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    for _ in range(3):
        assert load_prompt(prompt, digest) == ("grade the claims", digest)
    assert len(reads) == 1


def test_a_calibration_run_without_its_calibration_file_is_refused(tmp_path):
    cfg = gen_config(tmp_path, calibration_run=tmp_path / "no_calibration")
    with pytest.raises(UserError, match=r"has no calibration\.json"):
        uncalibrated_metrics(cfg, "opus")


def test_a_prompt_path_that_is_no_file_is_refused(tmp_path):
    for path in (tmp_path / "absent.md", tmp_path):
        with pytest.raises(UserError, match=r"not found|is a directory"):
            load_prompt(path, expected_sha256=None)


def test_a_prompt_file_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "prompt.md"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(UserError, match="is not UTF-8 text"):
        load_prompt(path, expected_sha256=None)

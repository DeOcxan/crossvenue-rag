"""The inputs a user gives outside options and config keys: files, environment, preconditions.

A suite-level test (code-style rule 27): each table feeds every input of one group the
out-of-domain values of the input inventory and checks the refusal names the input.
Command options are tested in `tests/cli/test_app.py`, config keys in the tests of the
config section modules (`tests/test_config_pipeline.py` and its siblings).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from crossvenue_rag.errors import UserError

# The file every case of the file table writes, and what it names in the refusal.
FILE = "input.data"


def _jsonl(path: Path) -> object:
    from crossvenue_rag import jsonl

    return jsonl.read(path)


def _questions(path: Path) -> object:
    from crossvenue_rag.dataset.questions import load_questions

    return load_questions(path)


def _answers(path: Path) -> object:
    from crossvenue_rag.generation.records import read_answers

    return read_answers(path)


def _yaml(path: Path) -> object:
    from crossvenue_rag.yaml_files import read_mapping

    return read_mapping(path, hint="fix it")


def _sources(path: Path) -> object:
    from crossvenue_rag.corpus.sources import load_sources

    return load_sources(path)


def _manifest(path: Path) -> object:
    from crossvenue_rag.corpus.manifest import read_manifest

    return read_manifest(path)


def _sample(path: Path) -> object:
    from crossvenue_rag.generation.calibration_sample import read_sample

    return read_sample(path)


def _id_map(path: Path) -> object:
    from crossvenue_rag.index.migrate import IdMap

    return IdMap.load(path)


def _old_chunks(path: Path) -> object:
    from crossvenue_rag.index.migrate import build_id_map

    return build_id_map(path, [], to_index_version="")


def _run_index(path: Path) -> object:
    from crossvenue_rag.run_files import RunIndex, read_json_model

    return read_json_model(path, RunIndex)


def _per_query(path: Path) -> object:
    from crossvenue_rag.evaluation.results import PerQueryRow
    from crossvenue_rag.run_files import read_table

    return read_table(path, PerQueryRow)


QUESTION = {"id": "cv-1", "question": "q one", "tier": "gold", "venues": ["bybit"], "qtype": "fact"}
MANIFEST_HEADER = "venue,venue_type,source_url,local_path,fetched_at,commit_sha,sha256\n"

# (reader, the file's bytes or None for no file or "<dir>" for a directory, what the refusal names)
FILE_OUT_OF_DOMAIN: list[tuple[Callable[[Path], object], str | bytes | None, str]] = [
    (_jsonl, None, "not found"),
    (_jsonl, "<dir>", "is a directory where a file is expected"),
    (_jsonl, '{"a": 1}\n{"a": \n', f"{FILE}:2: not JSON"),
    (_jsonl, '{"a": 1}\n[1, 2]\n', f"{FILE}:2: a JSON list where an object is expected"),
    (_jsonl, b'{"a": "\xff"}\n', f"{FILE}:1: not UTF-8"),
    (_questions, "<dir>", "is a directory where a file is expected"),
    (_questions, json.dumps({**QUESTION, "venues": "bybit"}) + "\n", f"{FILE}:1: venues"),
    (_questions, json.dumps({**QUESTION, "tier": "platinum"}) + "\n", f"{FILE}:1: tier"),
    (_questions, json.dumps({**QUESTION, "is_answerable": "no"}) + "\n", "is_answerable"),
    (_questions, json.dumps({"id": "cv-1", "question": "q"}) + "\n", f"{FILE}:1: "),
    (_questions, (json.dumps(QUESTION) + "\n") * 2, f"{FILE}:2: id cv-1 is also on line 1"),
    (_answers, json.dumps({"answer_id": "a"}) + "\n", f"{FILE}:1: "),
    (_yaml, None, "not found"),
    (_yaml, "a: [\n", "is not valid YAML"),
    (_yaml, "- a\n- b\n", "holds a list at the top level"),
    (_sources, None, "not found"),
    (_sources, "sources:\n  - binance\n", f"{FILE}: sources.0"),
    (_manifest, "venue,sha256\nbinance,x\n", "has no column"),
    (_manifest, MANIFEST_HEADER + "binance,cefi,u,p,t,c,s\n", f"{FILE}:2: venue_type"),
    (_manifest, MANIFEST_HEADER + "binance,cex,u\n", f"{FILE}:2: the row has fewer cells"),
    (_sample, None, "not found"),
    (_sample, "cv-1  fact\n", "names no run"),
    (_sample, "# 1 answers from results/x, stratified\n", "names no answer"),
    (_id_map, "{not json", "Invalid JSON"),
    (_id_map, json.dumps({"old_ids": 1}), "map: Field required"),
    (_old_chunks, json.dumps({"chunk_id": "c1"}) + "\n", f"{FILE}:1: the record lacks a string"),
    (_run_index, "{", f"{FILE}: Invalid JSON"),
    (_run_index, json.dumps({"index_version": "v", "nosuch": 1}), "nosuch"),
    (_per_query, "question_id,method\ncv-1,bm25\n", f"{FILE}:2"),
]  # fmt: skip


@pytest.mark.parametrize(("read", "content", "names"), FILE_OUT_OF_DOMAIN)
def test_a_file_outside_its_shape_is_refused_naming_the_file(tmp_path, read, content, names):
    path = tmp_path / FILE
    if content == "<dir>":
        path.mkdir()
    elif isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(UserError) as refused:
        read(path)
    assert names in str(refused.value)
    assert FILE in str(refused.value)


def test_a_remap_refuses_what_is_not_a_list_of_ids_and_ids_no_record_has(tmp_path):
    from crossvenue_rag.index.migrate import IdMap, remap_jsonl

    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps({"id": "q1", "gold_chunk_ids": "c1"}) + "\n", encoding="utf-8")
    ids = IdMap({"c1": "n1"}, old_chunks_sha256="", to_index_version="")
    with pytest.raises(UserError, match="gold_chunk_ids is not a list of chunk ids"):
        remap_jsonl(source, tmp_path / "out.jsonl", ids, ["gold_chunk_ids"], only=None)
    source.write_text(json.dumps({"id": "q1", "gold_chunk_ids": ["c1"]}) + "\n", encoding="utf-8")
    with pytest.raises(UserError, match="has the id q9"):
        remap_jsonl(source, tmp_path / "out.jsonl", ids, ["gold_chunk_ids"], only={"q9"})
    assert not (tmp_path / "out.jsonl").exists()


def _service_config(monkeypatch, value: str) -> object:
    from fastapi.testclient import TestClient

    from crossvenue_rag.service.app import create_app

    monkeypatch.setenv("CROSSVENUE_CONFIG", value)
    with TestClient(create_app()):
        return None


def _dsn(monkeypatch, value: str) -> object:
    from crossvenue_rag.config_loading import load_config

    monkeypatch.setenv("CROSSVENUE_PG_DSN", value)
    return load_config("configs/default.yaml")


def _key(monkeypatch, value: str) -> object:
    from crossvenue_rag.generation import api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", value)
    monkeypatch.setattr(api_key, "DOTENV_FILE", Path("/nonexistent/dotenv"))
    return api_key.load_api_key()


# (the variable's reader, a value outside its domain, what the refusal names)
ENV_OUT_OF_DOMAIN = [
    (_service_config, "/nonexistent/config.yaml", "CROSSVENUE_CONFIG"),
    (_service_config, "", "CROSSVENUE_CONFIG"),
    (_dsn, "notadsn", "store.dsn"),
    (_key, "   ", "OPENROUTER_API_KEY is not set"),
]


@pytest.mark.parametrize(("read", "value", "names"), ENV_OUT_OF_DOMAIN)
def test_an_environment_variable_outside_its_domain_is_refused_naming_it(
    monkeypatch, read, value, names
):
    with pytest.raises(UserError) as refused:
        read(monkeypatch, value)
    assert names in f"{refused.value} {refused.value.hint}"


@pytest.mark.parametrize(
    ("value", "debug"), [("", False), ("0", False), ("false", False), ("No", False), ("1", True)]
)
def test_crossvenue_debug_is_off_for_empty_0_false_and_no(monkeypatch, value, debug):
    from crossvenue_rag.cli.errors import debug_requested

    monkeypatch.setenv("CROSSVENUE_DEBUG", value)
    assert debug_requested() is debug


def _power_design(**changed: object) -> object:
    from crossvenue_rag.evaluation.power import PowerDesign

    fields: dict[str, object] = {
        "magnitudes": [0.5], "psi": 0.5, "deltas": [0.1], "n": 10, "alpha": 0.05,
        "family": 2, "studies": 5, "permutations": 5, "seed": 1,
    }  # fmt: skip
    return PowerDesign(**{**fields, **changed})  # type: ignore[arg-type]


def _preconditions() -> list[tuple[str, Callable[[], object], str]]:
    from crossvenue_rag.evaluation import bench, mcnemar_power, metrics, paired, t_test_size
    from crossvenue_rag.generation.calibration import select_judge
    from crossvenue_rag.generation.calibration_sample import calibration_sample
    from crossvenue_rag.reporting.compute.prompts import smallest_p
    from crossvenue_rag.retrieval.filters import resolve_venues
    from crossvenue_rag.stats import percentile

    return [
        ("size", lambda: mcnemar_power.mcnemar_power(0, 0.2, 0.1), "at least one question"),
        ("psi", lambda: mcnemar_power.mcnemar_power(20, 0.0, 0.0), "psi"),
        ("delta > psi", lambda: mcnemar_power.mcnemar_power(20, 0.2, 0.5), "delta must lie"),
        ("delta 0", lambda: mcnemar_power.questions_needed(0.3, 0.0), "a delta of 0"),
        ("delta > psi", lambda: mcnemar_power.questions_needed(0.3, 0.4), "delta must lie"),
        ("split", lambda: mcnemar_power.expected_split(20, 0.2, 0.5), "delta must lie"),
        ("effect", lambda: t_test_size.questions_for_effect(0.0), "effect size"),
        ("t-test size", lambda: t_test_size.detectable_effect(0), "at least one question"),
        ("McNemar split", lambda: paired.mcnemar_p(12, 10), "cannot favor one side 12"),
        ("Holm p", lambda: paired.holm([math.nan, 0.01]), "p-values lie in [0, 1]"),
        ("permutations", lambda: paired.paired_randomisation([1], [0], 0, 0), "permutations"),
        ("resamples", lambda: paired.paired_bootstrap([1], [0], 0, 0), "resamples"),
        ("studies", lambda: _power_design(studies=0), "studies must be at least 1"),
        ("permutations", lambda: _power_design(permutations=0), "permutations must be"),
        ("magnitudes", lambda: _power_design(magnitudes=[]), "difference magnitude"),
        ("bench rows", lambda: bench.summarise_modes([]), "no bench row for mode"),
        ("fraction", lambda: percentile([1.0], 1.5), "fraction lies in [0, 1]"),
        ("changed", lambda: smallest_p(0), "at least one changed question"),
        ("recall k", lambda: metrics.recall_at_k(["a"], ["a"], 0), "k must be positive"),
        ("nDCG k", lambda: metrics.ndcg_at_k(["a"], ["a"], 0), "k must be positive"),
        ("sample n", lambda: calibration_sample([], 0, 1, exclude_questions=()), "n >= 1"),
        ("candidates", lambda: select_judge({}, {}, {}), "at least one candidate"),
        ("venue None", lambda: resolve_venues({"venue": [None]}, {"binance": "cex"}), "None"),
    ]


@pytest.mark.parametrize("index", range(len(_preconditions())))
def test_a_library_precondition_is_a_value_error_naming_it(index):
    name, call, names = _preconditions()[index]
    with pytest.raises(ValueError) as refused:
        call()
    assert not isinstance(refused.value, UserError), name
    assert names in str(refused.value), name


def test_the_detectable_difference_stays_within_the_models_ceiling():
    """The grid ends at psi: past it a question would favor a method with probability above 1."""
    from crossvenue_rag.evaluation.mcnemar_power import detectable_delta

    found = detectable_delta(20, 0.2)
    assert math.isnan(found) or found <= 0.2

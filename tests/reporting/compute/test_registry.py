"""Tests for `reporting.compute.registry`."""

from __future__ import annotations

import json

from crossvenue_rag.reporting.compute import registry
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import registry_json_of
from tests.reporting.conftest import write_file


def test_runs_are_checked_against_the_current_index(tmp_path):
    runs = {"a": {"index_version": "v1"}, "b": {"index_version": "v0"}, "c": {}}
    write_file(tmp_path, "results/runs.json", registry_json_of(runs))
    write_file(tmp_path, "data/index/index_meta.json", json.dumps({"index_version": "v1"}))
    src = Sources(tmp_path)
    assert registry.run_index_version(src, "a") == "v1"
    assert registry.run_index_version(src, "c") is None
    assert registry.runs_on_current_index(src, ("a",))
    assert not registry.runs_on_current_index(src, ("a", "b"))
    assert not registry.runs_on_current_index(src, ("missing",))

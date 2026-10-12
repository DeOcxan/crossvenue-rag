"""Tests for `reporting.compute.repository`: the repository's own files, sized."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.models import ChunkSource
from crossvenue_rag.reporting.compute import repository
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import write_file


def source(venue, doc_path):
    return ChunkSource(venue, doc_path, ["Limits"], f"https://example.invalid/{doc_path}", 0)


def test_source_lines_count_every_line_of_every_py_file(tmp_path):
    write_file(tmp_path, "src/a.py", "x = 1\n\n")
    write_file(tmp_path, "src/pkg/b.py", "# comment\ny = 2\nz = 3\n")
    write_file(tmp_path, "src/notes.txt", "not code\n")
    assert repository.source_lines(Sources(tmp_path)) == 5


def test_the_stated_source_lines_are_read_from_the_text():
    assert repository.stated_source_lines("The cost is about 20,000 lines under `src/`.") == 20000
    assert repository.stated_source_lines("No size is stated.") == 0


def test_source_lines_agree_only_at_the_nearest_thousand():
    assert repository.source_lines_agree(22000, 21688)
    assert repository.source_lines_agree(20000, 20000)
    assert repository.source_lines_agree(21000, 20500)
    assert not repository.source_lines_agree(20000, 21688)
    assert not repository.source_lines_agree(28000, 30662)
    # "about 20,001 lines" is no round size.
    assert not repository.source_lines_agree(20001, 20001)


def test_subpackages_are_directories_with_an_init(tmp_path):
    for name in ("a/__init__.py", "b/__init__.py", "c/module.py", "d.py"):
        write_file(tmp_path, f"src/crossvenue_rag/{name}", "")
    assert repository.subpackage_count(Sources(tmp_path)) == 2


def test_import_contracts_are_counted_in_pyproject(tmp_path):
    contract = '[[tool.importlinter.contracts]]\nname = "{}"\ntype = "layers"\n'
    write_file(tmp_path, "pyproject.toml", "".join(contract.format(i) for i in range(3)))
    assert repository.import_contract_count(Sources(tmp_path)) == 3


def test_fixture_pages_are_the_keys_of_the_test_corpus(tmp_path):
    conftest = (
        "CORPUS: dict[str, tuple[str, str, str]] = {\n"
        '    "binance/rest-api.md": (\n        "binance",\n    ),\n'
        '    "bybit/rate-limit.md": (\n        "bybit",\n    ),\n'
        "}\n"
    )
    write_file(tmp_path, "tests/conftest.py", "import pytest\n\n" + conftest)
    assert repository.fixture_pages(Sources(tmp_path)) == 2


def test_the_python_requirement_is_read_from_pyproject(tmp_path):
    write_file(tmp_path, "pyproject.toml", '[project]\nrequires-python = ">=3.12"\n')
    assert repository.python_requirement(Sources(tmp_path)) == "3.12"
    write_file(tmp_path, "pyproject.toml", "[project]\n")
    assert repository.python_requirement(Sources(tmp_path)) == "?"
    # Any valid TOML spelling of the key is read, single quotes too.
    write_file(tmp_path, "pyproject.toml", "[project]\nrequires-python='>=3.13'\n")
    assert repository.python_requirement(Sources(tmp_path)) == "3.13"


def test_directions_are_the_second_level_headings(tmp_path):
    text = "# Directions\n\n## One\n\n### Detail\n\n## Two\n"
    write_file(tmp_path, "docs/explanation/directions.md", text)
    assert repository.direction_count(Sources(tmp_path)) == 2


def test_pytest_count_sums_the_tests_collected_per_file(tmp_path):
    write_file(tmp_path, "pyproject.toml", '[tool.pytest.ini_options]\naddopts = "-q"\n')
    write_file(
        tmp_path, "tests/test_a.py", "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n"
    )
    write_file(tmp_path, "tests/test_b.py", "def test_three():\n    pass\n")
    assert repository.pytest_count(str(tmp_path), every=False) == 3


def test_pytest_count_counts_the_filtered_out_tests_only_with_every(tmp_path):
    options = '[tool.pytest.ini_options]\naddopts = "-q -m \'not slow\'"\nmarkers = ["slow: s"]\n'
    write_file(tmp_path, "pyproject.toml", options)
    test = (
        "import pytest\n\n\ndef test_fast():\n    pass\n\n\n"
        "@pytest.mark.slow\ndef test_slow():\n    pass\n"
    )
    write_file(tmp_path, "tests/test_a.py", test)
    assert repository.pytest_count(str(tmp_path), every=False) == 1
    assert repository.pytest_count(str(tmp_path), every=True) == 2


def test_pytest_count_fails_when_nothing_is_collected(tmp_path):
    write_file(tmp_path, "pyproject.toml", '[tool.pytest.ini_options]\naddopts = "-q"\n')
    with pytest.raises(RuntimeError, match="no collected tests"):
        repository.pytest_count(str(tmp_path), every=False)


def test_pytest_count_fails_with_a_value_error_on_a_tests_line_without_a_count(tmp_path):
    # `-s` lets the test module's output reach the collected list.
    write_file(tmp_path, "pyproject.toml", '[tool.pytest.ini_options]\naddopts = "-q -s"\n')
    write_file(
        tmp_path, "tests/test_a.py", 'print("tests/no-count")\n\n\ndef test_a():\n    pass\n'
    )
    with pytest.raises(ValueError, match="invalid literal"):
        repository.pytest_count(str(tmp_path), every=False)


def test_suite_level_test_files_are_those_named_after_no_module(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/labeling/judge.py", "")
    write_file(tmp_path, "tests/labeling/test_judge.py", "")
    write_file(tmp_path, "tests/labeling/test_session.py", "")
    write_file(tmp_path, "tests/test_pipeline_smoke.py", "")
    write_file(tmp_path, "scripts/figures.py", "")
    write_file(tmp_path, "tests/scripts/test_figures.py", "")
    write_file(tmp_path, "tests/scripts/test_plots.py", "")
    assert repository.suite_level_test_files(Sources(tmp_path)) == [
        "tests/labeling/test_session.py",
        "tests/scripts/test_plots.py",
        "tests/test_pipeline_smoke.py",
    ]


def test_code_lines_leave_out_blank_lines_comments_and_docstrings():
    text = (
        '"""Module.\n\nMore."""\n\n# a comment\nx = 1\n\n\n'
        'def f():\n    """Doc."""\n    return (\n        x\n    )\n'
    )
    assert repository.code_lines(text) == 5


def test_the_largest_module_is_the_one_with_the_most_code_lines(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/a.py", '"""Doc."""\nx = 1\n')
    write_file(tmp_path, "src/crossvenue_rag/sub/b.py", "x = 1\ny = 2\n# no\n")
    assert repository.largest_module(Sources(tmp_path)) == ("src/crossvenue_rag/sub/b.py", 2)


def test_the_largest_module_is_searched_in_tests_and_scripts_too(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/a.py", "x = 1\n")
    write_file(tmp_path, "tests/test_a.py", "x = 1\ny = 2\n")
    write_file(tmp_path, "scripts/s.py", "x = 1\ny = 2\nz = 3\n")
    assert repository.largest_module(Sources(tmp_path)) == ("scripts/s.py", 3)


def test_a_pyproject_without_import_contracts_is_a_user_error_naming_the_fix(tmp_path):
    write_file(tmp_path, "pyproject.toml", '[tool.importlinter]\nroot_package = "a"\n')
    with pytest.raises(UserError, match=r"pyproject\.toml sets no list") as e:
        repository.import_contract_count(Sources(tmp_path))
    assert "[[tool.importlinter.contracts]]" in e.value.hint


def test_a_pyproject_that_is_not_valid_toml_is_a_user_error_naming_the_fix(tmp_path):
    write_file(tmp_path, "pyproject.toml", "[tool.importlinter\ncontracts = []\n")
    with pytest.raises(UserError, match=r"pyproject\.toml is not valid TOML") as e:
        repository.import_contract_count(Sources(tmp_path))
    assert "line and column" in e.value.hint


def test_a_pyproject_that_is_not_utf8_is_a_user_error(tmp_path):
    (tmp_path / "pyproject.toml").write_bytes(b'[project]\nrequires-python = "\xff"\n')
    with pytest.raises(UserError, match=r"pyproject\.toml is not UTF-8 text"):
        repository.python_requirement(Sources(tmp_path))

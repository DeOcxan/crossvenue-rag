"""Tests for `cli.reporting`: runs-manifest writes the registry make reproduce checks."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from crossvenue_rag.cli import app


def test_runs_manifest_lists_committed_runs_and_names_the_unversioned(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("results/*\n!results/a/\n!results/b/\n")
    for name, version in (("a", "v1"), ("b", None)):
        run = tmp_path / "results" / name
        run.mkdir(parents=True)
        if version:
            (run / "index.json").write_text(json.dumps({"index_version": version}))
        (run / "report.md").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["runs-manifest"])
    assert result.exit_code == 0, result.output
    assert "2 runs -> results/runs.json" in result.output
    assert "no index_version b" in result.output
    registry = json.loads((tmp_path / "results" / "runs.json").read_text())
    assert registry["runs"]["a"]["index_version"] == "v1"
    assert set(registry["runs"]["b"]["files"]) == {"report.md"}


def _fake_reproduce(*, problems: list[str]):
    from types import SimpleNamespace

    def reproduce(root):
        return SimpleNamespace(
            problems=problems,
            compared=[SimpleNamespace(path="README.md", line=2, text="7", reason="{} runs")],
            identifiers=[SimpleNamespace(path="README.md", line=4, text="abc", reason="at {}")],
            given=[SimpleNamespace(path="README.md", line=3, text="42", reason="by hand")],
            unsaved=[SimpleNamespace(path="docs/x.md", line=7, text="0.5", reason="no run")],
            run_cells=4,
            checked=9,
            refs=2,
        )

    return reproduce


def test_reproduce_summarises_and_lists_every_number_with_its_kind_when_verbose(monkeypatch):
    import crossvenue_rag.reporting.reproduce as reproduce

    monkeypatch.setattr(reproduce, "reproduce", _fake_reproduce(problems=[]))
    result = CliRunner().invoke(app, ["reproduce", "--verbose"])
    out = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "compared README.md:2: 7 ({} runs)" in out
    assert "identifier README.md:4: abc (at {})" in out
    assert "given README.md:3: 42 (by hand)" in out
    assert "unsaved docs/x.md:7: '0.5' (no run)" in out
    assert "4 cells" in out and "9 values compared (table cells included)" in out
    assert "2 identifiers checked" in out and "0 problems" in out


def test_reproduce_lists_only_the_unsaved_numbers_without_verbose(monkeypatch):
    import crossvenue_rag.reporting.reproduce as reproduce

    monkeypatch.setattr(reproduce, "reproduce", _fake_reproduce(problems=[]))
    out = CliRunner().invoke(app, ["reproduce"]).output
    kinds = {line.split()[0] for line in out.splitlines() if line.strip()}
    assert "unsaved" in kinds
    assert not kinds & {"compared", "identifier", "given"}


def test_reproduce_exits_1_on_a_problem_and_hides_given_numbers(monkeypatch):
    import crossvenue_rag.reporting.reproduce as reproduce

    monkeypatch.setattr(reproduce, "reproduce", _fake_reproduce(problems=["README.md:1: 3 != 4"]))
    result = CliRunner().invoke(app, ["reproduce"])
    assert result.exit_code == 1
    assert "README.md:1: 3 != 4" in result.output and "1 problem" in result.output
    assert "given " not in result.output


def test_the_verbose_help_names_the_details_and_says_unsaved_ones_always_print():
    from typing import get_type_hints

    from crossvenue_rag.cli.reporting import reproduce_command

    params = get_type_hints(reproduce_command, include_extras=True)["verbose"].__metadata__[0]
    assert params.help.startswith("Details of the check")
    assert "the unsaved ones always print" in params.help


def test_the_reproduce_summary_prints_its_counts_with_thousands_separators(monkeypatch):
    from types import SimpleNamespace

    report = SimpleNamespace(
        problems=[], given=[], unsaved=[], run_cells=15882, checked=1674, refs=1203
    )
    monkeypatch.setattr("crossvenue_rag.reporting.reproduce.reproduce", lambda root: report)
    result = CliRunner().invoke(app, ["reproduce"])
    out = " ".join(result.output.split())
    assert "15,882 cells of saved run tables" in out
    assert "1,674 values compared (table cells included), 1,203 identifiers checked" in out
    assert "0 problems" in out


def test_reproduce_prints_bracketed_document_text_as_it_is(monkeypatch):
    import crossvenue_rag.reporting.reproduce as reproduce

    cited = "docs/design.md:1: text not found: 'Cormack et al. 2009, [doi:10.1145/1571941.1572114]'"
    tag = "README.md:2: does not hold: a [/bold] b"
    monkeypatch.setattr(reproduce, "reproduce", _fake_reproduce(problems=[cited, tag]))
    result = CliRunner().invoke(app, ["reproduce"])
    out = " ".join(result.output.split())
    assert result.exit_code == 1, result.output
    assert "[doi:10.1145/1571941.1572114]" in out
    assert "a [/bold] b" in out and "2 problems" in out


def test_runs_manifest_prints_a_bracketed_run_name_as_it_is(tmp_path, monkeypatch):
    from types import SimpleNamespace

    written: list = []
    registry = SimpleNamespace(runs={"[/x]": SimpleNamespace(index_version=None)})
    monkeypatch.setattr("crossvenue_rag.cli.reporting.build_registry", lambda root: registry)
    monkeypatch.setattr(
        "crossvenue_rag.cli.reporting.write_registry", lambda path, r: written.append(path)
    )
    (tmp_path / ".gitignore").write_text("results/*\n")
    (tmp_path / "results").mkdir()
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["runs-manifest"])
    assert result.exit_code == 0, (result.exception, result.output)
    assert "no index_version [/x]" in result.output
    assert len(written) == 1 and not (tmp_path / "results" / "runs.json").exists()

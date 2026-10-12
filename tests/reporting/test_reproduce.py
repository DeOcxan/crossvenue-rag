"""`make reproduce` on a fixture run, and the documents it checks.

Nothing here reads the repository's own results: every run, dataset and document is
built under tmp_path, except the list of the repository's Markdown files.
"""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting import reproduce as reproduce_module
from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.readme import CHECKED_NUMBERS_TEMPLATE
from crossvenue_rag.reporting.compute.answers import outcome_count
from crossvenue_rag.reporting.reproduce import document_specs, reproduce
from crossvenue_rag.reporting.runs import build_registry
from crossvenue_rag.reporting.sources import Sources
from crossvenue_rag.run_files import RUNS_REGISTRY, write_registry
from tests.reporting.conftest import CONFIGS, write_fixture

ROOT = Path(__file__).resolve().parents[2]
# The Markdown documents `crossvenue reproduce` does not check, each with the reason.
# docs/style.md, Checks: a number outside reproduce is re-derived from its source by hand.
BY_HAND = "not under reproduce: its numbers are re-derived by hand"
UNCHECKED = {
    "docs/README.md": "the index of the documentation",
    "docs/adr/README.md": "the index of the decision records",
    "docs/style.md": "rules for Markdown: its numbers are rule numbers and examples",
    "docs/code-style.md": "rules for Python: its numbers are rule numbers and examples",
    "evals/datasets/CHANGELOG.md": "the history of the dataset versions, whose entries stay",
    "docs/explanation/directions.md": BY_HAND,
    "docs/how-to/label.md": BY_HAND,
    "docs/how-to/reproduce.md": BY_HAND,
    "evals/README.md": BY_HAND,
}


def _refusal_spec(doc: Doc, src: Sources) -> None:
    rows = {
        "comparison": [
            f"{outcome_count(src, c, 'comparison', 'refused', 1)} to "
            f"{outcome_count(src, c, 'comparison', 'refused', 2)}"
            for c in CONFIGS
        ]
    }
    doc.table("Refusals, v1 to v2:", rows)
    partial = [str(outcome_count(src, c, None, "partial", 2)) for c in CONFIGS]
    doc.check("Partial answers under v2: {}, {}, {} and {}.", *partial)


DOC = """# Findings

Refusals, v1 to v2:

| category | gemini@venue_split | qwen@venue_split | mercury@venue_split | gemini@hybrid_rerank |
|---|---:|---:|---:|---:|
| comparison | 1 to 0 | 0 to 0 | 0 to {mercury} | 0 to 0 |

Partial answers under v2: 1, 0, {partial} and 0.
"""


def write_uncommitted_fixture(root: Path) -> None:
    """The fixture runs, none of them committed, with the empty registry that matches."""
    write_fixture(root)
    (root / ".gitignore").write_text("", encoding="utf-8")
    write_registry(root / RUNS_REGISTRY, build_registry(root))


def only_the_refusal_spec(monkeypatch):
    """`reproduce` checks findings.md alone, with `_refusal_spec`."""
    monkeypatch.setattr(
        reproduce_module, "document_specs", lambda root, count: {"findings.md": _refusal_spec}
    )


def test_reproduce_passes_on_a_fixture_run_that_matches_its_document(tmp_path, monkeypatch):
    only_the_refusal_spec(monkeypatch)
    write_uncommitted_fixture(tmp_path)
    (tmp_path / "findings.md").write_text(DOC.format(mercury=1, partial=0))
    report = reproduce(tmp_path)
    assert report.problems == []
    assert report.checked == 8


def test_reproduce_reports_where_the_document_disagrees_with_the_run(tmp_path, monkeypatch):
    only_the_refusal_spec(monkeypatch)
    write_uncommitted_fixture(tmp_path)
    # The document counts the claim-less partial answer as partial; the outcome rule
    # counts it as refused.
    (tmp_path / "findings.md").write_text(DOC.format(mercury=0, partial=1))
    report = reproduce(tmp_path)
    assert [str(p) for p in report.problems] == [
        "findings.md:7: row 'comparison', column 4: '0 to 0' in the text, '0 to 1' recomputed",
        "findings.md:9: 1 in the text, 0 recomputed (hole 3 of "
        "'Partial answers under v2: {}, {}, {} and {}.')",
    ]


def with_the_readme_count(monkeypatch, stated: str) -> None:
    """`reproduce` checks findings.md with `_refusal_spec` and a README stating `stated`."""

    def count_spec(doc: Doc, src: Sources) -> None:
        doc.given(CHECKED_NUMBERS_TEMPLATE, stated, reason="the total")

    monkeypatch.setattr(reproduce_module, "CHECKED_NUMBERS", stated)
    monkeypatch.setattr(
        reproduce_module,
        "document_specs",
        lambda root, count: {"findings.md": _refusal_spec, "README.md": count_spec},
    )


def test_reproduce_accepts_the_readme_count_that_equals_its_total(tmp_path, monkeypatch):
    with_the_readme_count(monkeypatch, "8")
    write_uncommitted_fixture(tmp_path)
    (tmp_path / "findings.md").write_text(DOC.format(mercury=1, partial=0))
    (tmp_path / "README.md").write_text(f"It {CHECKED_NUMBERS_TEMPLATE.format('8')}. All.\n")
    report = reproduce(tmp_path)
    assert report.problems == []
    assert report.checked == 8


def test_reproduce_reports_a_readme_count_that_differs_from_its_total(tmp_path, monkeypatch):
    with_the_readme_count(monkeypatch, "9")
    write_uncommitted_fixture(tmp_path)
    (tmp_path / "findings.md").write_text(DOC.format(mercury=1, partial=0))
    (tmp_path / "README.md").write_text(f"It {CHECKED_NUMBERS_TEMPLATE.format('9')}. All.\n")
    report = reproduce(tmp_path)
    assert [str(p) for p in report.problems] == [
        "README.md:1: 9 in the text, 8 recomputed (values compared, every document)"
    ]


def markdown_documents(root: Path) -> set[str]:
    """README.md and every Markdown file under docs/ and evals/, relative to `root`."""
    found = {"README.md"} if (root / "README.md").exists() else set()
    for tree in ("docs", "evals"):
        found |= {str(p.relative_to(root)) for p in (root / tree).rglob("*.md")}
    return found


def test_every_markdown_document_is_checked_by_reproduce_or_named_unchecked():
    # A document added without a spec in `document_specs` states numbers no check
    # recomputes; it is named here until it gets a spec or a reason in `UNCHECKED`.
    checked = set(document_specs(ROOT, lambda *, every: 0))
    assert markdown_documents(ROOT) - checked == set(UNCHECKED)
    assert checked <= markdown_documents(ROOT)

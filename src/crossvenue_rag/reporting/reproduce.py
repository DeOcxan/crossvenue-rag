"""`make reproduce`: the numbers of the documents' text, checked against their sources.

The documents and the function that states each one's claims are in `document_specs`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import formatting
from ..config import SCHEMA_DEFAULTS
from ..evaluation.results import PER_QUERY_FILE
from ..run_files import RUNS_REGISTRY, SUMMARY_FILE
from .check import Doc
from .claims.adr import adr
from .claims.agreement import agreement_doc
from .claims.cli_reference import cli_reference
from .claims.config_reference import config_reference
from .claims.corpus_datasheet import corpus_datasheet
from .claims.datasheet import datasheet
from .claims.design import design
from .claims.domain_research import domain_research
from .claims.findings import findings
from .claims.methodology import methodology
from .claims.readme import CHECKED_NUMBERS, CHECKED_NUMBERS_TEMPLATE, TestCounter, readme
from .compute.repository import pytest_count
from .outcomes import Listed, Problem
from .rederive import rederive_retrieval_run
from .runs import committed_runs, verify_registry
from .sources import Sources

FINDINGS = "docs/results/findings.md"
README = "README.md"
METHODOLOGY = "evals/methodology.md"
DESIGN = "docs/explanation/design.md"
CONFIG_REFERENCE = "docs/reference/config.md"
CLI_REFERENCE = "docs/reference/cli.md"
CORPUS_DATASHEET = "docs/reference/corpus.md"
DATASHEET = "evals/datasheet.md"
DOMAIN_RESEARCH = "evals/domain-research.md"
AGREEMENT = "evals/agreement.md"


def adr_paths(root: str | Path) -> list[str]:
    """Every decision record except the index."""
    return sorted(
        str(p.relative_to(root))
        for p in (Path(root) / "docs/adr").glob("[0-9][0-9][0-9][0-9]-*.md")
    )


Spec = Callable[[Doc, Sources], None]


def document_specs(root: str | Path, test_count: TestCounter) -> dict[str, Spec]:
    """Each checked document and the function that states its claims.

    `test_count` counts the tests README.md states (`claims.readme.TestCounter`).
    """
    return {
        FINDINGS: findings,
        README: lambda doc, s: readme(doc, s, test_count),
        METHODOLOGY: methodology,
        DESIGN: design,
        **{path: adr for path in adr_paths(root)},
        CONFIG_REFERENCE: config_reference,
        CLI_REFERENCE: cli_reference,
        CORPUS_DATASHEET: corpus_datasheet,
        DATASHEET: datasheet,
        DOMAIN_RESEARCH: domain_research,
        AGREEMENT: agreement_doc,
    }


@dataclass
class Report:
    """The result of `reproduce`: the checked documents and the problems of the saved runs."""

    docs: list[Doc] = field(default_factory=list)
    run_problems: list[Problem] = field(default_factory=list)
    run_cells: int = 0  # cells of saved tables re-derived from per_query.csv

    @property
    def problems(self) -> list[Problem]:
        """Every problem: the saved runs' first, then each document's."""
        return self.run_problems + [p for d in self.docs for p in d.problems]

    @property
    def given(self) -> list[Listed]:
        """The numbers every document declares as given."""
        return [g for d in self.docs for g in d.given_numbers]

    @property
    def unsaved(self) -> list[Listed]:
        """The numbers every document declares as unsaved."""
        return [u for d in self.docs for u in d.unsaved_numbers]

    @property
    def compared(self) -> list[Listed]:
        """The values every document's checks compared, table cells included."""
        return [c for d in self.docs for c in d.compared_numbers]

    @property
    def identifiers(self) -> list[Listed]:
        """The identifiers every document's checks compared."""
        return [i for d in self.docs for i in d.identifiers]

    @property
    def checked(self) -> int:
        """How many values the documents' checks compared, table cells included."""
        return sum(d.checked for d in self.docs)

    @property
    def refs(self) -> int:
        """How many identifiers (run ids, index versions, hashes) the documents' checks compared."""
        return sum(d.refs for d in self.docs)


def reproduce(root: str | Path) -> Report:
    """Checks the saved runs, then runs each document's spec and audits it.

    The runs come first: every committed run's files must match the sha256 in
    `results/runs.json`. Every retrieval run whose question file is unchanged must have
    metric tables equal to what the evaluation code re-derives from its per_query.csv
    (`reporting.rederive`). A number read from such a run's aggregate file is then one the
    evaluator would write again. A run whose question file changed, or that records none,
    is skipped.

    Each document's claims are then checked and the document audited (`Doc.audit`). The
    report holds the problems and the values the checks listed, with their kind.
    """
    src = Sources(root)
    specs = document_specs(root, lambda every: pytest_count(str(root), every=every))
    report = Report()
    root_path = Path(root)
    report.run_problems += [Problem(str(RUNS_REGISTRY), 0, m) for m in verify_registry(root_path)]
    for name in committed_runs(root_path):
        run = SCHEMA_DEFAULTS.paths.results / name
        run_dir = root_path / run
        if (run_dir / SUMMARY_FILE).exists() and (run_dir / PER_QUERY_FILE).exists():
            cells, problems = rederive_retrieval_run(run_dir, root_path)
            report.run_cells += cells
            report.run_problems += [Problem(str(run), 0, m) for m in problems]
    for path, spec in specs.items():
        doc = Doc.read(path, root)
        spec(doc, src)
        doc.audit()
        report.docs.append(doc)
    _check_total(report)
    return report


def _check_total(report: Report) -> None:
    """README.md's count of the values compared, against the total of every document.

    No document's claims can recompute the total, which only exists once every document
    has been checked; the README's claims give it, and this compares it.
    """
    stated = next((d for d in report.docs if d.path == README), None)
    total = formatting.count(report.checked)
    if stated is None or total == CHECKED_NUMBERS:
        return
    at = stated.text.find(CHECKED_NUMBERS_TEMPLATE.split("{}")[0])
    stated.problems.append(
        Problem(
            README,
            stated.line_of(max(at, 0)),
            f"{CHECKED_NUMBERS} in the text, {total} recomputed (values compared, every document)",
        )
    )

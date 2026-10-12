"""Reporting commands: reproduce, runs-manifest.

`reporting.reproduce` is imported inside its command: the claims load scipy and
scikit-learn, which `crossvenue --help` does not need. Every other import is at the top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from .. import formatting
from ..config import SCHEMA_DEFAULTS
from ..errors import UserError
from ..evaluation.results import PER_QUERY_FILE
from ..reporting.runs import GITIGNORE, build_registry
from ..run_files import RUNS_REGISTRY, write_registry
from .app import console, say


def reproduce_command(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help=(
                "Details of the check: the values it lists, with their kind; "
                "the unsaved ones always print."
            ),
        ),
    ] = False,
) -> None:
    """Checks the committed runs, then the checked documents' numbers against their sources.

    The checked documents are those reporting/reproduce.py lists.
    """
    from ..reporting.reproduce import reproduce

    _require_repository_root()
    report = reproduce(".")
    # Printed as text: a problem quotes document text, and a citation such as
    # `[doi:...]` would be read by rich as a style tag.
    for problem in report.problems:
        say(console, str(problem), style="red", soft_wrap=True)
    if verbose:
        kinds = (
            ("compared", report.compared),
            ("identifier", report.identifiers),
            ("given", report.given),
        )
        for kind, listed in kinds:
            for n in listed:
                where = formatting.location(n.path, n.line)
                say(console, f"{kind} {where}: {n.text} ({n.reason})", soft_wrap=True)
    for u in report.unsaved:
        say(
            console,
            Text.assemble(
                ("unsaved", "yellow"),
                f" {formatting.location(u.path, u.line)}: '{u.text}' ({u.reason})",
            ),
            soft_wrap=True,
        )
    say(
        console,
        f"{formatting.plural(report.run_cells, 'cell')} of saved run tables re-derived from "
        f"{PER_QUERY_FILE}; "
        f"{formatting.plural(report.checked, 'value')} compared (table cells included), "
        f"{formatting.plural(report.refs, 'identifier')} checked, "
        f"{formatting.count(len(report.given))} given, "
        f"{formatting.count(len(report.unsaved))} unsaved, "
        f"{formatting.plural(len(report.problems), 'problem')}",
    )
    if report.problems:
        raise typer.Exit(1)


def runs_manifest_command() -> None:
    """Writes results/runs.json: every committed run, its file hashes and `index_version`.

    It runs after a run is committed (its `!results/<run>/` line in
    .gitignore); `make reproduce` then fails if any listed file changes.
    """
    _require_repository_root()
    registry = build_registry(Path("."))
    write_registry(RUNS_REGISTRY, registry)
    unversioned = [n for n, e in registry.runs.items() if e.index_version is None]
    say(console, f"{formatting.plural(len(registry.runs), 'run')} -> {RUNS_REGISTRY}")
    for name in unversioned:
        say(console, Text("no index_version", style="yellow"), name)


def _require_repository_root() -> None:
    """Refuses to run outside the repository: both commands read `.gitignore` and `results/`.

    `results/` is `paths.results`, the directory of the runs committed to the repository
    (`reporting.runs.committed_runs`).

    Raises:
        UserError: The working directory has no `.gitignore` or no `results/`.
    """
    results = SCHEMA_DEFAULTS.paths.results
    if not (Path(GITIGNORE).is_file() and results.is_dir()):
        raise UserError(
            f"{Path.cwd()} is not the repository root: it has no {GITIGNORE} and {results}/",
            hint="run the command from the root of the crossvenue-rag checkout",
        )

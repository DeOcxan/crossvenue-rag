"""Corpus commands: fetch, ingest, stats, count-term, check."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.text import Text

from .. import formatting
from ..config import Config
from ..config_loading import DEFAULT_CONFIG
from ..corpus.artifacts import count_term as run_count
from ..corpus.artifacts import read_corpus_stats
from ..corpus.fetch import fetch_all
from ..corpus.ingest import ingest as run_ingest
from ..index.status import build_status
from ..index.store import StoreState
from . import render
from .app import ConfigOpt, SetOpt, console, load, say

# The width of the name column in `check`'s list of artifacts.
CHECK_NAME_WIDTH = 12


def fetch(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    venue: Annotated[
        list[str] | None, typer.Option("--venue", help="Venues to fetch; repeatable.")
    ] = None,
    follow_branch: Annotated[
        bool,
        typer.Option(
            "--follow-branch",
            help="Each git source at its branch head, with the manifest rewritten.",
        ),
    ] = False,
    allow_missing: Annotated[
        bool,
        typer.Option(
            "--allow-missing", help="Continuing past a GitBook page that cannot be downloaded."
        ),
    ] = False,
) -> None:
    """Downloads the documentation sources the manifest records and verifies them.

    By default each git source is checked out at the commit data/manifest.csv
    records and every file is checked against its sha256; GitBook pages are
    compared the same way; the manifest is not rewritten. `--follow-branch`
    fetches the branch heads instead and rewrites the manifest.
    """
    cfg = load(config, set_)
    report = fetch_all(cfg, only=venue, follow_branch=follow_branch, allow_missing=allow_missing)
    say(console, render.fetch_summary(report.rows))
    if report.manifest_written:
        say(console, f"manifest rewritten: {cfg.paths.manifest}")
    else:
        say(
            console,
            f"every file matches {cfg.paths.manifest} "
            f"({formatting.plural(sum(report.matched.values()), 'document')}); "
            "the manifest is unchanged",
        )


def ingest(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Normalizes and chunks everything in the manifest."""
    cfg = load(config, set_)
    chunks = run_ingest(cfg)
    say(
        console,
        Text(formatting.plural(len(chunks), "chunk"), style="green"),
        f"-> {cfg.paths.processed}",
    )
    _print_stats(cfg)


def stats(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Prints corpus statistics from the last ingestion."""
    _print_stats(load(config, set_))


def _print_stats(cfg: Config) -> None:

    stats = read_corpus_stats(cfg)
    say(console, render.corpus_stats_table(stats))
    say(console, render.chunk_length_histogram(stats))
    if stats.dedup is not None:
        say(console, render.dedup_summary(stats.dedup))
    if stats.metadata_coverage is not None:
        say(console, render.metadata_coverage(stats.metadata_coverage))
    if stats.index_version is not None:
        say(console, f"index_version (of this ingest): {stats.index_version}")


def count_term(
    term: Annotated[str, typer.Argument(help="Substring to look for.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    venue: Annotated[str | None, typer.Option("--venue", help="The one venue to count in.")] = None,
) -> None:
    """Counts chunks containing a term, by the project's one counting convention.

    Chunks containing the term, case-insensitive, over data/processed/chunks.jsonl.
    Every count of one term quoted in a document or a `notes` field is this
    command's output, so two people counting the same thing agree.
    """
    cfg = load(config, set_)
    counts = run_count(cfg, term, venue)
    say(console, render.term_counts(term, counts, venue))


def check(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Validates the config and reports what is built."""
    cfg = load(config, set_)
    status = build_status(cfg)
    say(console, Text.assemble(("config ok", "green"), f": {config}"))
    say(
        console,
        Text.assemble(("sources ok", "green"), f": {formatting.plural(status.venues, 'venue')}"),
    )
    for artifact in status.artifacts:
        mark = ("present", "green") if artifact.present else ("missing", "yellow")
        say(
            console,
            Text.assemble(f"{artifact.name:{CHECK_NAME_WIDTH}s} ", mark, f"  {artifact.path}"),
        )
    if isinstance(status.store, StoreState):
        state = status.store
        say(
            console,
            Text.assemble(
                f"{'pgvector':{CHECK_NAME_WIDTH}s} ",
                ("connected", "green"),
                f"  {formatting.plural(state.rows, 'row')}, models {', '.join(state.models)}, "
                f"index_version {', '.join(state.index_versions)}",
            ),
        )
    else:
        say(
            console,
            Text.assemble(
                f"{'pgvector':{CHECK_NAME_WIDTH}s} ", ("unavailable", "yellow"), f"  {status.store}"
            ),
        )
    if status.index_version is None:
        mark, detail = ("inconsistent", "yellow"), f"  {status.inconsistency}"
    else:
        mark, detail = ("consistent", "green"), f"  {status.index_version} ({status.agreeing})"
    say(console, Text.assemble(f"{'index':{CHECK_NAME_WIDTH}s} ", mark, detail))

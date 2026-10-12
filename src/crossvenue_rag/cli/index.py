"""Index commands: index, id-map, remap-ids."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from .. import formatting
from ..config_loading import DEFAULT_CONFIG
from ..corpus.artifacts import CHUNKS_FILENAME
from ..dataset.remap import RemapRequest, remap_file
from ..errors import UserError
from ..index.build import build_indexes
from ..index.migrate import write_id_map
from ..index_version import index_meta_path, require_built_index_version
from .app import ConfigOpt, SetOpt, console, load, say


def index(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    skip_dense: Annotated[
        bool, typer.Option("--skip-dense", help="The BM25 index alone, with no dense index.")
    ] = False,
) -> None:
    """Builds the pgvector and BM25 indexes from chunks.jsonl."""
    cfg = load(config, set_)
    built = build_indexes(cfg, skip_dense=skip_dense)
    say(
        console,
        Text.assemble(
            ("BM25", "green"), f": {formatting.plural(built.chunks, 'chunk')} -> {built.bm25_dir}"
        ),
    )
    if built.dense_rows is None:
        console.print(
            "[yellow]skipped dense index[/yellow]: eval and label will refuse to run "
            "until Postgres is rebuilt at the same index_version"
        )
        return
    say(
        console,
        Text.assemble(
            ("pgvector", "green"),
            f": {formatting.plural(built.dense_rows, 'chunk')} -> {cfg.store.table}",
        ),
    )
    say(console, Text.assemble("index_version: ", (built.index_version, "bold")))


def id_map(
    old_chunks: Annotated[Path, typer.Argument(help=f"{CHUNKS_FILENAME} from before the rebuild.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Path of the map file.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Maps every old chunk id onto the current chunk with the same venue and text."""
    cfg = load(config, set_)
    if out.is_dir():
        raise UserError(f"--out {out} is a directory", hint="pass --out the path of the map file")
    mapping = write_id_map(cfg, old_chunks, out)
    say(
        console,
        Text.assemble(
            (formatting.count(len(mapping.mapping)), "green"),
            f" old ids -> {formatting.count(mapping.canonical_ids)} canonical ids: {out}",
        ),
    )


def remap_ids(
    map_path: Annotated[Path, typer.Option("--map", help="Map written by `id-map`.")],
    source: Annotated[Path, typer.Argument(help="JSONL dataset, or a .rounds.json journal.")],
    target: Annotated[
        Path | None,
        typer.Argument(
            help="Output path (JSONL); none for a journal, which is rewritten in place."
        ),
    ] = None,
    field: Annotated[
        list[str] | None,
        typer.Option(
            "--field",
            help="Id-list field to rewrite, JSONL only (default: `source_chunk_ids` and "
            "`gold_chunk_ids`); repeatable.",
        ),
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Record ids to rewrite; the others are kept as they are."),
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
) -> None:
    """Carries chunk-id pointers through an id map, verifying nothing else changed."""
    cfg = load(config, set_)
    # A field left out is None, so a journal can refuse a --field it would ignore.
    request = RemapRequest(
        map_path, source, target, tuple(field) if field else None, tuple(only or ())
    )
    report, written = remap_file(request, require_built_index_version(index_meta_path(cfg)))
    say(
        console,
        f"{formatting.plural(report.records, 'record')}, "
        f"{formatting.count(report.records_changed)} changed; ids "
        f"{formatting.count(report.ids_before)} -> {formatting.count(report.ids_after)}; "
        "unmapped (left as-is): "
        f"{formatting.count(len(report.unmapped))} -> {written}",
    )
    for chunk_id in sorted(report.unmapped):
        say(console, Text.assemble("  ", ("unmapped", "yellow"), f" {chunk_id}"))

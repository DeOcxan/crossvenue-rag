"""What every run kind shares: its directory, index.json, config.yaml, CSV tables, the registry.

One writer or reader each, so every run directory, index record, saved config and
CSV is made or read the same way. Every CSV has the dialect of the committed runs: the
csv module's defaults, CRLF line ends and minimal quoting.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from . import formatting
from .clock import run_id
from .config import SCHEMA_DEFAULTS, Config
from .errors import UserError, invalid_file, require_file
from .yaml_files import read_mapping

# The file names more than one run kind writes.
INDEX_FILE = "index.json"
CONFIG_FILE = "config.yaml"
SUMMARY_FILE = "summary.csv"
PAIRED_FILE = "paired.csv"
REPORT_FILE = "report.md"
SOURCE_FILE = "source.json"
# The registry of committed runs, relative to the repository root.
RUNS_REGISTRY = SCHEMA_DEFAULTS.paths.results / "runs.json"
# How much of a file is hashed at a time: 1 MiB, so a large file is never read whole.
HASH_BLOCK_BYTES = 1 << 20
# What a damaged run file is told to do: crossvenue writes every one of them.
RUN_FILE_HINT = "restore the file from the run, or make the run again"


class RunSource(BaseModel):
    """source.json: the runs a judged or prompts run was made from.

    Two formats: a judged run's `source_run`, `judge`, `stops` and, in runs
    that record their spend, `spent_usd`; a prompts run's `v1` and `v2`. A field a format
    lacks is None and is not written.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_run: str | None = None
    judge: str | None = None
    stops: list[str] | None = None
    spent_usd: float | None = None
    v1: str | None = None
    v2: str | None = None

    def runs(self) -> list[str]:
        """The source runs it names: `v1`, `v2`, then `source_run`."""
        return [r for r in (self.v1, self.v2, self.source_run) if r is not None]


def write_run_source(run_dir: Path, source: RunSource) -> None:
    """Writes `source` as the run's source.json, two-space indented, without a final newline."""
    text = json.dumps(source.model_dump(exclude_none=True), indent=2)
    (run_dir / SOURCE_FILE).write_text(text, encoding="utf-8")


def read_json_model[M: BaseModel](path: Path, model: type[M], hint: str = RUN_FILE_HINT) -> M:
    """A JSON file crossvenue wrote, validated as `model`.

    Raises:
        UserError: Nothing is at `path`, or a directory is. The file is not UTF-8, not JSON, or not
            a valid `model`; the error names the file and the field, `hint` the fix.
    """
    source = require_file(path, hint=hint)
    try:
        return model.model_validate_json(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise UserError(f"{path} is not UTF-8 text", hint=hint) from exc
    except ValidationError as exc:
        raise invalid_file(path, exc, hint=hint) from exc


def read_run_source(run_dir: Path) -> RunSource | None:
    """The run's source.json (`RunSource` lists its formats); None when the run has none."""
    path = run_dir / SOURCE_FILE
    if not path.exists():
        return None
    return read_json_model(path, RunSource)


def run_path(results_dir: Path, suffix: str = "") -> Path:
    """A run's directory, named by its start time: `results_dir/<run id><suffix>`.

    The caller creates it.
    """
    return results_dir / (run_id() + suffix)


def create_run_dir(run_dir: Path) -> Path:
    """Creates `run_dir` and returns it; a run never writes into a directory that exists.

    Raises:
        UserError: `run_dir` exists, as when two runs of one kind start in the same second.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise UserError(
            f"run directory {run_dir} exists already",
            hint="start the run again: run ids are UTC time to the second",
        ) from exc
    return run_dir


def write_run_config(run_dir: Path, cfg: Config) -> None:
    """The run's resolved config as `config.yaml` (`Config.to_yaml`).

    Every key is written except `store.dsn`, which can hold a password.
    """
    (run_dir / CONFIG_FILE).write_text(cfg.to_yaml(), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """The hex sha256 of the file at `path`, read in blocks of `HASH_BLOCK_BYTES`.

    The hash a run records for its question file and the manifest for each document.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def question_fields(questions: Path) -> dict[str, str | None]:
    """The question file a run read and its sha256; `None` when the file is gone."""
    return {
        "questions": str(questions),
        "questions_sha256": file_sha256(questions) if questions.exists() else None,
    }


class RunIndex(BaseModel):
    """index.json: what a run was made against.

    Formats of the committed files, each a subset of these fields in this order:

    - retrieval runs from 20260924T234623Z on: `index_version`, `questions`,
      `questions_sha256`, no final newline;
    - the pool run (`results/20260925T003721Z_pool`): `index_version`, `purpose`,
      `questions`, `questions_sha256`, `methods`, `k`, `category`, `questions_run`,
      no final newline;
    - the bench run (`results/20260925T225850Z_bench`): `index_version`, `questions`,
      no final newline;
    - answer, judged, prompts and power runs as `write_index` writes them, a format no
      committed run of those kinds has: `index_version`, and the question file for
      answer, judged and power runs, with a final newline.

    Runs saved before their kind recorded an index have none: the retrieval run
    20260924T200525Z, the answer, judged and prompts runs of 2026-09-25 and 2026-09-26,
    and every power run before 2026-09-27; `generation.runs` then falls back to
    `results/runs.json`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_version: str
    purpose: str | None = None
    questions: str | None = None
    questions_sha256: str | None = None
    methods: list[str] | None = None
    k: int | None = None
    category: str | None = None
    questions_run: int | None = None


def index_json(index: RunIndex, *, final_newline: bool) -> str:
    """index.json as written: the fields set, in order, indented by two."""
    fields = index.model_dump(mode="json", exclude_unset=True)
    return json.dumps(fields, indent=2) + ("\n" if final_newline else "")


def write_index(run_dir: Path, index: RunIndex | Mapping[str, Any], *, final_newline: bool) -> None:
    """Writes `index.json` (`RunIndex`); a mapping is validated as one, its fields in order.

    Retrieval, pool and bench runs were committed without a final newline and
    answer and prompts runs end with one; `final_newline` keeps each as it is.
    """
    record = index if isinstance(index, RunIndex) else RunIndex.model_validate(dict(index))
    (run_dir / INDEX_FILE).write_text(index_json(record, final_newline=final_newline), "utf-8")


def read_index(run_dir: Path) -> RunIndex | None:
    """A run's `index.json` (`RunIndex` lists its formats), or `None` for a run that has none."""
    path = run_dir / INDEX_FILE
    if not path.exists():
        return None
    return read_json_model(path, RunIndex)


class RegistryEntry(BaseModel):
    """One committed run in results/runs.json.

    Its `index_version`, where it comes from, its notes and the sha256 of each of its files.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_version: str | None
    index_version_from: str | None
    notes: list[str]
    files: dict[str, str]


class Registry(BaseModel):
    """results/runs.json: every committed run, by name.

    One format: `{"runs": {name: entry}}`, each entry's fields in `RegistryEntry`'s
    order, indented by two, with a final newline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: dict[str, RegistryEntry]


def read_registry(path: Path) -> Registry:
    """The registry of committed runs at `path`."""
    return read_json_model(path, Registry)


def registry_json(registry: Registry) -> str:
    """results/runs.json as written: indented by two, with a final newline."""
    return json.dumps(registry.model_dump(mode="json"), indent=2) + "\n"


def write_registry(path: Path, registry: Registry) -> None:
    """Writes the registry of committed runs to `path`."""
    path.write_text(registry_json(registry), encoding="utf-8")


def read_run_config(run_dir: Path) -> Config:
    """The config a run was made with: its `config.yaml`, validated as `Config`.

    Only the file is read; `CROSSVENUE_PG_DSN` does not apply. Every committed
    run's file validates. The runs up to 2026-09-26 record no model revisions
    (`embedding.revision`, `reranker.revision`) and come in five formats, which differ
    in the other keys they lack. `Config` fills a missing key with its current default,
    so a check reads only keys its run saved:

    - the retrieval runs 20260924T200525Z and 20260924T234623Z: no `generation`
      and no `eval` bootstrap or paired-comparison keys;
    - the retrieval runs of 2026-09-25 up to 20260925T024237Z and the pool run
      20260925T003721Z_pool: the bootstrap and paired-comparison keys added;
    - 20260925T132613Z and 20260925T144223Z: `eval.primary_comparison` added;
    - the answer runs 20260925T175128Z_answers and 20260925T202827Z_answers and
      the bench run 20260925T225850Z_bench: `generation` added, without the
      calibration thresholds, `calibration_run`, `calibration_exclude_behaviour`
      and the prompt hashes;
    - the judged runs of 2026-09-26: without `calibration_run` and the prompt
      hashes;
    - the retrieval runs of 2026-10-10, 20261010T214504Z and 20261010T222553Z: the
      whole config, the model revisions included.

    `Config.to_yaml` writes every key but `store.dsn`, so a run saved later holds the
    whole config of its day without the connection string, and the schema default fills
    it. A retired key, which the schema does not have, is dropped
    before validation (`RETIRED_CONFIG_KEYS`: `eval.score_duplicates_as_equivalent`, in
    20260924T200525Z); any other unknown key is refused, as in a user config.

    Raises:
        UserError: The schema rejects a key or a value of config.yaml.
    """
    path = run_dir / CONFIG_FILE
    raw = read_mapping(path, hint=RUN_FILE_HINT)
    for key in RETIRED_CONFIG_KEYS:
        _drop(raw, key.split("."))
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise invalid_file(path, exc, hint=RUN_FILE_HINT) from exc


# Config keys that committed runs saved and the schema does not have.
RETIRED_CONFIG_KEYS = ("eval.score_duplicates_as_equivalent",)


def _drop(node: dict[str, object], path: Sequence[str]) -> None:
    *parents, last = path
    for key in parents:
        child = node.get(key)
        if not isinstance(child, dict):
            return
        node = child
    node.pop(last, None)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """A CSV table: the header row, then one row per sequence."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _blank_is_none(value: object) -> object:
    return None if value == "" else value


# A numeric cell that may be empty: an empty cell reads as None and None writes as empty.
OptionalInt = Annotated[int | None, BeforeValidator(_blank_is_none)]
OptionalFloat = Annotated[float | None, BeforeValidator(_blank_is_none)]
OptionalBool = Annotated[bool | None, BeforeValidator(_blank_is_none)]


class CsvRow(BaseModel):
    """A row of a saved CSV table.

    A subclass types the columns every format of its table has or may have. A
    column it does not declare, such as a metric at a cutoff (`recall@10`), is
    kept as its cell and read through `number` or `text`.
    """

    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    def cells(self) -> dict[str, Any]:
        """Every column of the row by its saved name, declared ones first."""
        return self.model_dump(by_alias=True)

    def text(self, column: str) -> str:
        """The cell of `column` as text, empty for a declared column the file lacks.

        Raises:
            ValueError: `column` is neither declared nor in the file; the message names it.
        """
        name = _field_names(type(self)).get(column)
        extra = self.model_extra or {}
        if name is None and column not in extra:
            raise ValueError(f"the row has no column {column}")
        value = getattr(self, name) if name is not None else extra[column]
        return "" if value is None else str(value)

    def number(self, column: str) -> float | None:
        """The cell of `column` as a number, None when empty.

        Raises:
            ValueError: `column` is neither declared nor in the file.
        """
        cell = self.text(column)
        return float(cell) if cell else None

    def value(self, column: str) -> float:
        """The cell of `column` as a number.

        Raises:
            ValueError: `column` is neither declared nor in the file, or its cell is empty.
        """
        number = self.number(column)
        if number is None:
            raise ValueError(f"the {column} cell is empty")
        return number

    def count(self, column: str) -> int:
        """The cell of `column` as a count.

        Raises:
            ValueError: `column` is neither declared nor in the file, or its cell is empty or
                holds a fraction.
        """
        value = self.value(column)
        if not value.is_integer():
            raise ValueError(
                f"the {column} cell holds {formatting.own_form(value)}, which is no whole number"
            )
        return int(value)


@cache
def _field_names(row_type: type[CsvRow]) -> dict[str, str]:
    """The declared fields of `row_type` by their saved column name."""
    return {field.alias or name: name for name, field in row_type.model_fields.items()}


@dataclass(frozen=True)
class CsvTable[RowT: CsvRow]:
    """A saved CSV table: its columns in their saved order, and its rows."""

    columns: tuple[str, ...]
    rows: tuple[RowT, ...]

    @classmethod
    def of(cls, rows: Sequence[RowT]) -> CsvTable[RowT]:
        """The rows under the columns of the first one; no column without a row."""
        return cls(tuple(rows[0].cells()) if rows else (), tuple(rows))


def read_table[RowT: CsvRow](path: Path, row_type: type[RowT]) -> CsvTable[RowT]:
    """The CSV table at `path` with each row validated as `row_type`.

    An empty file is a table without columns or rows.

    Raises:
        UserError: Nothing is at `path`, or a directory is. The file is not UTF-8, or a row is not a
            valid `row_type` (a missing column, a cell of the wrong type, a ragged row). The error
            names the file and the line.
    """
    with require_file(path, hint=RUN_FILE_HINT).open(encoding="utf-8", newline="") as fh:
        # A short row reads as None cells and a long one as cells under the key None.
        reader = csv.DictReader(fh)
        rows: list[RowT] = []
        try:
            for row in reader:
                if None in row or None in row.values():
                    named = [v for name, v in row.items() if name is not None and v is not None]
                    cells = len(named) + len(row.get(None) or [])
                    raise UserError(
                        f"{formatting.location(path, reader.line_num)}: a row of "
                        f"{formatting.plural(cells, 'cell')} under "
                        f"{formatting.plural(len(reader.fieldnames or ()), 'column')}",
                        hint=RUN_FILE_HINT,
                    )
                rows.append(row_type.model_validate(row))
        except UnicodeDecodeError as exc:
            raise UserError(f"{path} is not UTF-8 text", hint=RUN_FILE_HINT) from exc
        except ValidationError as exc:
            raise invalid_file(
                formatting.location(path, reader.line_num), exc, hint=RUN_FILE_HINT
            ) from exc
        return CsvTable(tuple(reader.fieldnames or ()), tuple(rows))


def write_table(path: Path, table: CsvTable[Any]) -> None:
    """Writes `table` in the shared dialect; an empty file when it has no column."""
    if not table.columns:
        path.write_text("", encoding="utf-8")
        return
    cells = (row.cells() for row in table.rows)
    write_csv(path, table.columns, ([c.get(name) for name in table.columns] for c in cells))

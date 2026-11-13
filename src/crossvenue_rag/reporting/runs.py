"""The registry of committed runs: what each was made against and its file hashes.

`crossvenue runs-manifest` writes the registry, `results/runs.json`, each time a run is
committed, with each run's `notes` from `run_notes.NOTES`. `make reproduce` (`verify_registry`)
checks that the registry lists exactly the committed runs, with each file's sha256 and each
run's `index_version`; it does not check the `notes`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import SCHEMA_DEFAULTS
from ..evaluation.power import POWER_FILE, read_power_study
from ..evaluation.results import (
    PER_QUERY_FILE,
    read_per_query,
)
from ..generation.records import ANSWERS_FILE, AnswerRecord, read_answers
from ..run_files import (
    INDEX_FILE,
    RUNS_REGISTRY,
    Registry,
    RegistryEntry,
    file_sha256,
    read_index,
    read_registry,
    read_run_source,
)
from .run_notes import NOTES

GITIGNORE = ".gitignore"


def committed_runs(root: Path) -> list[str]:
    """The runs committed to the repository: the `!results/<run>/` lines of `.gitignore`."""
    text = (root / GITIGNORE).read_text(encoding="utf-8")
    results = re.escape(SCHEMA_DEFAULTS.paths.results.as_posix())
    return re.findall(rf"^!{results}/([^/\s]+)/$", text, flags=re.M)


def file_hashes(run_dir: Path) -> dict[str, str]:
    """The sha256 of each file in `run_dir`, by file name."""
    return {path.name: file_sha256(path) for path in sorted(run_dir.iterdir()) if path.is_file()}


def _own_version(run_dir: Path) -> str | None:
    index = read_index(run_dir)
    return None if index is None else index.index_version


@dataclass(frozen=True)
class RetrievalTops:
    """A retrieval run's `index_version` and each (question, method)'s ranked chunk ids."""

    index_version: str
    top: Mapping[tuple[str, str], list[str]]


def load_retrieval_tops(retrieval_runs: Mapping[str, Path]) -> dict[str, RetrievalTops]:
    """Each retrieval run that recorded an `index_version`, read once."""
    tops = {}
    for name, run_dir in sorted(retrieval_runs.items()):
        version = _own_version(run_dir)
        if version is not None:
            rows = read_per_query(run_dir).rows
            tops[name] = RetrievalTops(version, {(r.question_id, r.method): r.ids for r in rows})
    return tops


def derive_answer_version(
    answers: Sequence[AnswerRecord], retrieval: Mapping[str, RetrievalTops]
) -> tuple[str, str] | None:
    """(`index_version`, retrieval run) of the first retrieval run, by name, the answers match.

    A run matches when every generated answer's chunk ids equal the top k of its question
    and method; None when no answer was generated or no run matches.
    """
    generated = [(a, a.generation) for a in answers if a.generation is not None]
    if not generated:
        return None
    for name, run in sorted(retrieval.items()):
        if all(
            run.top.get((a.question_id, a.retriever), [])[: len(g.chunk_ids)] == g.chunk_ids
            for a, g in generated
        ):
            return run.index_version, name
    return None


def build_registry(root: Path) -> Registry:
    """results/runs.json: every committed run's `index_version` and file hashes.

    For each run committed to the repository: its `index_version`, where it comes from, its `notes`
    and the sha256 of each of its files. The `index_version`, where the run has one, comes
    from one of:

    - `index.json`: the run recorded it;
    - derived: an answer or judged run saved before runs recorded a version is matched
      against the retrieval runs. Every answer's chunks equal the top k of the same question and
      method in a retrieval run made at that version;
    - a source run: a prompts or power run without an `index.json` inherits the version
      of the runs it was computed from.
    """
    results = root / SCHEMA_DEFAULTS.paths.results
    dirs = {name: results / name for name in committed_runs(root)}
    retrieval = load_retrieval_tops(
        {n: d for n, d in dirs.items() if (d / PER_QUERY_FILE).exists()}
    )
    versions = {name: _version(run_dir, retrieval) for name, run_dir in dirs.items()}
    # In registry order, so a run can inherit from a source run that itself inherited.
    for name, run_dir in dirs.items():
        if versions[name].index_version is None:
            versions[name] = _inherited_version(versions, _source_runs(run_dir)) or versions[name]
    return Registry(
        runs={
            name: RegistryEntry(
                index_version=versions[name].index_version,
                index_version_from=versions[name].source,
                notes=NOTES.get(name, []),
                files=file_hashes(run_dir),
            )
            for name, run_dir in dirs.items()
        }
    )


@dataclass(frozen=True)
class _Version:
    """A run's `index_version` and where it comes from; both None when it has none."""

    index_version: str | None = None
    source: str | None = None


def _version(run_dir: Path, retrieval: Mapping[str, RetrievalTops]) -> _Version:
    """A run's `index_version` from its own index.json, else derived from its answers."""
    own = _own_version(run_dir)
    if own is not None:
        return _Version(own, INDEX_FILE)
    if (run_dir / ANSWERS_FILE).exists():
        found = derive_answer_version(read_answers(run_dir / ANSWERS_FILE), retrieval)
        if found is not None:
            source = (SCHEMA_DEFAULTS.paths.results / found[1]).as_posix()
            return _Version(found[0], f"derived: chunks equal the top k of {source}")
    return _Version()


def _inherited_version(versions: Mapping[str, _Version], sources: list[str]) -> _Version | None:
    """The one `index_version` every source run has; None otherwise."""
    found = {versions.get(Path(s).name, _Version()).index_version for s in sources}
    if not sources or len(found) != 1 or None in found:
        return None
    return _Version(found.pop(), "source runs: " + ", ".join(sorted(sources)))


def _source_runs(run_dir: Path) -> list[str]:
    """The runs a derived run names: source.json's, or a power run's source run."""
    source = read_run_source(run_dir)
    if source is not None:
        return source.runs()
    if (run_dir / POWER_FILE).exists():
        return [read_power_study(run_dir).run]
    return []


def verify_registry(root: Path) -> list[str]:
    """The problems of the run registry against the committed runs.

    A missing registry, a committed run it does not list, a run it lists that is not
    committed, a file whose sha256 differs from it, and an `index_version` that differs from it.
    """
    path = root / RUNS_REGISTRY
    if not path.exists():
        return [f"{RUNS_REGISTRY} is missing (`crossvenue runs-manifest` writes it)"]
    recorded = read_registry(path).runs
    fresh = build_registry(root).runs
    problems = [
        f"{RUNS_REGISTRY}: {n} is committed but not listed" for n in fresh if n not in recorded
    ]
    problems += [
        f"{RUNS_REGISTRY}: {n} is listed but not committed" for n in recorded if n not in fresh
    ]
    for name in sorted(set(fresh) & set(recorded)):
        want, have = recorded[name], fresh[name]
        run = SCHEMA_DEFAULTS.paths.results / name
        for filename in sorted(set(want.files) | set(have.files)):
            if want.files.get(filename) != have.files.get(filename):
                problems.append(f"{run / filename}: sha256 differs from {RUNS_REGISTRY}")
        if want.index_version != have.index_version:
            problems.append(
                f"{run}: index_version {have.index_version} re-derived, "
                f"{want.index_version} in {RUNS_REGISTRY}"
            )
    return problems

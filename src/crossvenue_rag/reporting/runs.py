"""The registry of committed runs: what each was made against and its file hashes.

`crossvenue runs-manifest` writes the registry, `results/runs.json`, each time a run is
committed, with each run's notes from `NOTES`. `make reproduce` (`verify_registry`) checks
that the registry lists exactly the committed runs, with each file's sha256 and each run's
`index_version`; it does not check the notes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import formatting, listing
from ..config import SCHEMA_DEFAULTS
from ..evaluation.power import POWER_FILE, read_power_study
from ..evaluation.results import (
    PER_QUERY_FILE,
    read_per_query,
)
from ..generation.records import ANSWERS_FILE, AnswerRecord, read_answers
from ..generation.schemas import FIRST_SCHEMA
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

GITIGNORE = ".gitignore"
# The note of every retrieval run whose report.md calls the oracle an upper bound.
ORACLE_NOTE = (
    "report.md calls venue_split_oracle an upper bound without qualification. The oracle "
    "bounds only what better venue detection can add to venue coverage; it is no upper "
    "bound on recall, precision or nDCG (docs/adr/0007-venue-split.md). The sentence ends "
    '", not deployable.", the trailing form docs/style.md rule 26 bans.'
)
# Notes that no file of the run can carry, written into the registry.
VENUE_ATTRIBUTION_NOTE = (
    "report.md and paired.csv show venue_attribution_error with no mark; its judge dimension "
    "(venue_correct) passes calibration on the point estimate only, and methodology section 9 "
    "reports it as unresolved."
)
# The note of every retrieval run whose report.md footnote says more than unjudged@k and
# nDCG\u2032 can show.
POOLING_NOTE = (
    'The report.md footnote calls pooling bias "measured, not assumed" and, where it has '
    "nDCG\u2032, calls its gap to nDCG@k what the missing judgments cost. unjudged@k counts "
    "never-judged chunks and does not say how much they bias recall; the nDCG\u2032 gap is the "
    "change that dropping them makes, and a relevant chunk no pooled method retrieved is "
    "never judged, so neither counts it as relevant (evals/methodology.md, section 1)."
)
# The note of every retrieval run whose report.md lacks a disclosure column of section 1.
NO_UNJUDGED_NOTE = (
    "report.md has no unjudged@k or nDCG\u2032 column: the run predates the pooling-bias "
    "disclosure evals/methodology.md section 1 asks of every report."
)
NO_NDCG_PRIME_NOTE = (
    "report.md has no nDCG\u2032 column: the run predates it; results/20260925T015715Z "
    "reports it on the same retrievals."
)

# The note of every retrieval run whose report.md marks a missing value `n/a`.
MISSING_CELL_NOTE = (
    "report.md marks a metric a slice cannot have with n/a; docs/style.md rule 25 gives a "
    "table cell with no value a single dash, as the report template writes it."
)
# The note of every answer and judged run whose report.md lists the parameters as raw JSON.
RAW_PARAMS_NOTE = (
    "report.md lists the parameters sent as a raw JSON object; the report template writes "
    "them as key=value items, and summary.csv holds them as JSON."
)


# The note of every judged run whose report.md names its source run outside code font.
SOURCE_RUN_NOTE = (
    "report.md names its source run outside code font (docs/style.md rule 35); the report "
    "template writes it in code font."
)
# The British spellings the report.md of the retrieval runs 20260925T012314Z, 015715Z,
# 024237Z, 132613Z and 144223Z writes.
RETRIEVAL_SPELLINGS = ("favour", "judgement", "labelled", "labelling", "towards")
# The note of every run whose Markdown report writes a minus sign as an ASCII hyphen.
MINUS_NOTE = (
    "Its Markdown report writes every minus sign as an ASCII hyphen; docs/code-style.md "
    "rule 25 asks for U+2212, which formatting.py writes."
)
# The note of every retrieval and bench run whose report writes latencies at one decimal.
LATENCY_NOTE = (
    "Its Markdown report writes latencies in ms at one decimal and without a thousands "
    "separator; docs/style.md rule 11 gives a latency of 10 ms or more as an integer with a "
    "thousands separator."
)
# The note of every run whose Markdown report writes p-values at four decimals.
P_VALUE_NOTE = (
    "Its Markdown report writes p-values at four decimals; docs/style.md rule 11 gives them "
    "three, and a value below 0.001 as < 0.001."
)
# The note of every answer and judged run whose report.md writes USD and counts unformatted.
COST_COUNT_NOTE = (
    "report.md writes USD at four decimals, and token counts and latencies in ms with no "
    "thousands separator; docs/style.md rule 11 gives cost three decimals and counts and "
    "latencies a thousands separator."
)
# The note of every answer run whose report.md names its files outside code font.
FILE_NAMES_NOTE = (
    "report.md names per_answer.csv, summary.csv, paired.csv and answers.jsonl outside code "
    "font (docs/style.md rule 35)."
)
# The note of every judged run whose report.md names behaviour_correct outside code font.
JUDGED_IDENTIFIER_NOTE = (
    "report.md names the metric behaviour_correct in the paired rows outside code font "
    "(docs/style.md rules 20 and 35)."
)


def _spelling_note(*words: str) -> str:
    """The note of a run whose Markdown report writes `words` in British spelling."""
    return (
        f"Its Markdown report writes {listing.prose(list(words))} in British spelling; "
        "docs/style.md rule 20 asks for US English, which the report templates write."
    )


def _budget_note(spent: str) -> str:
    """The note of a judged run made with a `--budget` its config.yaml does not record."""
    return (
        "Made with a --budget override that config.yaml does not record: it shows "
        f"generation.budget_usd 3.0, while the judge calls it paid for cost {spent} "
        "(answers.jsonl, the uncached calls)."
    )


# The CPU-against-MPS difference of the retrieval runs on v0.2.1-gold, measured on 2026-10-10.
DEVICE_NOTE = (
    "Device effect: eval on v0.2.1-gold with embedding.device=cpu (20261010T222553Z) and the "
    "MPS run 20260925T012314Z differ in 25 of 684 retrieval lists. On the 50 comparison "
    "questions only nDCG and MRR differ, in the fourth decimal, and dense's unjudged@10 is "
    "0.086 on CPU against 0.078 on MPS. In other categories recall and precision differ too; "
    "the largest difference is version_sensitive hybrid recall@10, 0.717 on CPU against 0.750 "
    "on MPS. The registered results are from the MPS runs 20260925T012314Z and "
    "20260925T132613Z; the v0.2.2-gold sensitivity check pairs the two CPU runs "
    "20261010T222553Z and 20261010T214504Z."
)
# The record whose judgment a re-run from the cache supplies, found on 2026-10-10.
JUDGE_CACHE_NOTE = (
    "gemini@venue_split:cv-0002 records a judge error after 3 invalid responses. Its answer is "
    "identical to that of gemini@hybrid_rerank:cv-0002, so the judge request is the same, and "
    "this run cached a valid response to it when it judged gemini@hybrid_rerank:cv-0002 "
    "(2026-09-25T23:55:09Z, cache key "
    "0be2aa833b3f2bcfa8fe9f8b8207be81831336c9ce5cfd219c2762ea7012dc55). judge-run from the "
    "cache therefore judges "
    "gemini@venue_split:cv-0002 too, moving gemini@venue_split coverage:all from 0.5593 to "
    "0.5642 and behaviour_correct:all from 74/113 to 75/114. cv-0002 is a fact question, so "
    "no comparison-scope result changes, and no secondary result over all questions changes "
    "sign, significance or whether its interval excludes 0. The published numbers are those "
    "of this run."
)

NOTES: dict[str, list[str]] = {
    "20260924T200525Z": [
        "Made before exact deduplication and before runs recorded an index_version; "
        "its chunk ids are those of the pre-deduplication corpus.",
        NO_UNJUDGED_NOTE,
        "report.md tells the reader to run crossvenue label and then the evaluation again, "
        "an imperative that docs/style.md rule 19 allows only in a how-to procedure.",
        LATENCY_NOTE,
    ],
    "20260924T234623Z": [
        NO_UNJUDGED_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note("favour", "labelled", "towards"),
        LATENCY_NOTE,
    ],
    "20260925T002151Z": [
        ORACLE_NOTE,
        NO_UNJUDGED_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note("favour", "labelled", "labelling", "towards"),
        'report.md writes "--" as a dash in the footnote of the methods outside the labeling '
        "pool, which docs/code-style.md rule 25 bans.",
        MINUS_NOTE,
        LATENCY_NOTE,
    ],
    "20260925T003721Z_pool": [
        'POOL_RUN.md writes "--" as a dash, which docs/code-style.md rule 25 bans; the template '
        "writes a colon in its place."
    ],
    "20260925T012314Z": [
        ORACLE_NOTE,
        POOLING_NOTE,
        NO_NDCG_PRIME_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note(*RETRIEVAL_SPELLINGS),
        MINUS_NOTE,
        LATENCY_NOTE,
        DEVICE_NOTE,
    ],
    "20260925T015715Z": [
        ORACLE_NOTE,
        POOLING_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note(*RETRIEVAL_SPELLINGS),
        MINUS_NOTE,
        LATENCY_NOTE,
    ],
    "20260925T024237Z": [
        ORACLE_NOTE,
        POOLING_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note(*RETRIEVAL_SPELLINGS),
        MINUS_NOTE,
        LATENCY_NOTE,
    ],
    "20260925T132613Z": [
        ORACLE_NOTE,
        POOLING_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note(*RETRIEVAL_SPELLINGS, "randomisation"),
        MINUS_NOTE,
        P_VALUE_NOTE,
        LATENCY_NOTE,
    ],
    "20260925T144223Z": [
        ORACLE_NOTE,
        POOLING_NOTE,
        MISSING_CELL_NOTE,
        _spelling_note(*RETRIEVAL_SPELLINGS, "randomisation"),
        MINUS_NOTE,
        P_VALUE_NOTE,
        LATENCY_NOTE,
    ],
    "20260925T175128Z_answers": [
        "Generated with prompts/generator.v1.md (sha256 519bd2c2...) and answer schema "
        f"{formatting.own_form(FIRST_SCHEMA)}, as its config.yaml records; every answer's "
        "prompt_sha256 is the v1 hash.",
        RAW_PARAMS_NOTE,
        _spelling_note("behaviour", "randomisation"),
        "calibration.md names the run in its heading outside code font (docs/style.md rule "
        "35); the calibration template writes it in code font.",
        COST_COUNT_NOTE,
        FILE_NAMES_NOTE,
        "calibration.md names the metric behaviour_correct outside code font (docs/style.md "
        "rules 20 and 35), writes its intervals at two decimals beside values at three and "
        "USD per answer at four decimals (docs/style.md rule 11), and writes the minus sign "
        "of a negative bound as an ASCII hyphen (docs/code-style.md rule 25).",
    ],
    "20260925T202827Z_answers": [
        RAW_PARAMS_NOTE,
        _spelling_note("behaviour", "randomisation"),
        COST_COUNT_NOTE,
        FILE_NAMES_NOTE,
    ],
    "20260925T225850Z_bench": [
        "BENCH.md predates the formatting of its template: its counts carry hand-written "
        "plurals, and its load averages are a raw list and a raw dict; machine.json holds "
        "the load averages and the discarded pairs.",
        LATENCY_NOTE,
    ],
    "20260926T000001Z_judged": [
        "report.md and paired.csv predate the uncalibrated mark: faithfulness is the "
        "judge's grade only, its dimension (supported) failed calibration "
        "(results/20260925T175128Z_answers/calibration.json), and its rows support no "
        "conclusion.",
        VENUE_ATTRIBUTION_NOTE,
        _budget_note("$9.943"),
        RAW_PARAMS_NOTE,
        _spelling_note("behaviour", "randomisation"),
        SOURCE_RUN_NOTE,
        JUDGED_IDENTIFIER_NOTE,
        COST_COUNT_NOTE,
        MINUS_NOTE,
        P_VALUE_NOTE,
        JUDGE_CACHE_NOTE,
    ],
    "20260926T004836Z_judged": [
        "report.md and paired.csv predate the uncalibrated mark: faithfulness is the "
        "judge's grade only and supports no conclusion.",
        VENUE_ATTRIBUTION_NOTE,
        "paired.csv labels gemini@venue_split against gemini@hybrid_rerank primary; under "
        "prompt v2 that comparison is secondary and exploratory (methodology section 9a), "
        "and the primary comparison stays on the v1 run.",
        _budget_note("$11.695"),
        RAW_PARAMS_NOTE,
        _spelling_note("behaviour", "randomisation"),
        SOURCE_RUN_NOTE,
        JUDGED_IDENTIFIER_NOTE,
        COST_COUNT_NOTE,
        MINUS_NOTE,
        P_VALUE_NOTE,
    ],
    "20260926T004851Z_prompts": [
        _spelling_note("randomisation"),
        "report.md names the two judged runs in its heading outside code font (docs/style.md "
        "rule 35); the template writes them in code font.",
        MINUS_NOTE,
        P_VALUE_NOTE,
    ],
    "20261010T214504Z": [
        "Sensitivity run of the six retrieval methods on v0.2.2-gold, retrieval metrics only. "
        "Made with embedding.device=cpu: the MPS backend stalled in two attempts before any "
        "file was written. 20261010T222553Z is its baseline on v0.2.1-gold with the same "
        "config.",
        DEVICE_NOTE,
    ],
    "20261010T222553Z": [
        "The six retrieval methods on v0.2.1-gold with embedding.device=cpu, the baseline of "
        "the v0.2.2-gold sensitivity run 20261010T214504Z. On the comparison questions its "
        "recall, precision and venue_coverage equal those of 20260925T132613Z, made on MPS; "
        "nDCG and dense's MRR differ in the fourth decimal and dense's unjudged@10 is 0.086 "
        "against 0.078, from the device's effect on dense ranking.",
        DEVICE_NOTE,
    ],
}


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
    """(`index_version`, retrieval run) of the retrieval run the answers were generated from.

    That is the run where every generated answer's chunk ids equal the top k of its
    question and method; None when there is none.
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

    For each run committed to the repository: its `index_version`, where it comes from, its notes
    and the sha256 of each of its files. The `index_version` comes from one of:

    - `index.json`: the run recorded it;
    - derived: an answer run saved before runs recorded a version is matched against the
      retrieval runs. Every answer's chunks equal the top k of the same question and
      method in a retrieval run made at that version;
    - a source run: a prompts or power run inherits the version of the runs it was
      computed from.
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
        return [f"{RUNS_REGISTRY} is missing; write it with `crossvenue runs-manifest`"]
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

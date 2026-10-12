"""Dataset validation behind `crossvenue validate-evals`.

The checks catch what corrupts an evaluation without crashing it, such as a duplicated
question or a venue the corpus does not have.

Near-duplicate detection needs an encoder and is skipped with a warning when the encoder
cannot be loaded, so CI does not download the encoder's weights for it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import formatting
from ..config import Config
from ..corpus.artifacts import CHUNKS_FILENAME, chunks_path, load_chunk_venues, read_chunk_field
from ..corpus.manifest import read_manifest
from ..errors import UserError
from ..index.embeddings import Embedder
from ..index_version import built_index_version
from ..run_files import file_sha256, read_json_model
from .journal import read_journal, rounds_path
from .questions import (
    META_HINT,
    MIN_QUESTIONS_PER_CATEGORY,
    meta_path,
)
from .records import (
    DATASETS_DIR,
    DatasetMeta,
    QuestionRecord,
    as_question,
    load_records,
    manifest_fingerprint,
)

# One vector per text, row for row, as `Embedder.encode_documents` returns them.
DocumentEncoder = Callable[[list[str]], np.ndarray]

# The fewest questions that can hold a near-duplicate pair.
PAIR = 2


# The warning's threshold: the busiest venue named by more than this many times the
# questions of the quietest. It is a judgment for the report; no standard sets it.
LOPSIDED_VENUE_RATIO = 4


# evals/methodology.md section 6, check 2: a question sharing a run of this many words
# with one of its gold chunks restates the chunk instead of asking for it.
LEAKAGE_NGRAM = 6
# Quoted spans are exempt: an error string or identifier a developer pastes
# verbatim is a legitimate query term and is not counted as leakage.
_QUOTED_RE = re.compile(r"`[^`]*`|\"[^\"]*\"")
_WORD_RE = re.compile(r"[a-z0-9_]+")

# Question ids as they appear when one `notes` field refers to another question.
_QUESTION_ID_RE = re.compile(r"\bcv-\d{4}\b")


@dataclass(slots=True)
class ValidationStats:
    """The question counts of a validated set."""

    total: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_qtype: dict[str, int] = field(default_factory=dict)
    by_venue: dict[str, int] = field(default_factory=dict)
    answerable: int = 0
    unanswerable: int = 0
    # Questions with gold chunks; a question judged to have none is not counted.
    with_gold: int = 0


@dataclass(slots=True)
class ValidationReport:
    """The errors, warnings and statistics a dataset validation collects."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: ValidationStats = field(default_factory=ValidationStats)

    @property
    def ok(self) -> bool:
        """Whether the validation found no error; warnings do not count."""
        return not self.errors

    def error(self, message: str) -> None:
        """Records a problem that fails the validation."""
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Records a problem that is reported but does not fail the validation."""
        self.warnings.append(message)


def _check_ids(records: Sequence[QuestionRecord], report: ValidationReport) -> None:
    counts = Counter(r.id for r in records)
    for question_id, count in counts.items():
        if count > 1:
            report.error(f"duplicate id {question_id!r} appears {formatting.plural(count, 'time')}")


def _check_note_cross_references(
    records: Sequence[QuestionRecord], report: ValidationReport
) -> None:
    """Checks the question ids one `notes` field uses to point at another.

    Notes carry the reasoning a question rests on ("the answerable sibling is
    cv-0108", "this is the control for cv-0045"), and an annotator follows those
    pointers while judging. A dangling pointer, whose target does not exist, is
    an error. A pointer whose target became a different question when
    the set was renumbered after the note was written reads as correct and
    cannot be detected in general. Its common case, a note pointing at a
    question that shares no venue with it, is a warning, since a cross-venue
    reference can be legitimate.
    """
    known = {r.id: r for r in records}
    for record in records:
        for target in sorted(set(_QUESTION_ID_RE.findall(record.notes))):
            if target == record.id:
                report.warn(f"{record.id}: notes reference the question itself")
            elif target not in known:
                report.error(f"{record.id}: notes reference {target}, which is not in the dataset")
            elif (
                record.venues
                and known[target].venues
                and not set(record.venues) & set(known[target].venues)
            ):
                report.warn(
                    f"{record.id} (venues {', '.join(sorted(record.venues))}) references {target} "
                    f"(venues {', '.join(sorted(known[target].venues))}), which shares no venue "
                    "with it; check that the pointer still names the question it was written for"
                )


def _check_exact_duplicates(records: Sequence[QuestionRecord], report: ValidationReport) -> None:
    seen: dict[str, str] = {}
    for record in records:
        key = " ".join(record.question.lower().split())
        if key in seen:
            report.error(f"{record.id}: question text duplicates {seen[key]}")
        seen[key] = record.id


def _check_venues(
    records: Sequence[QuestionRecord], corpus_venues: set[str], report: ValidationReport
) -> None:
    for record in records:
        unknown = set(record.venues) - corpus_venues
        if unknown:
            # An unknown venue makes `--venue-filter` refuse the whole run, and
            # venue coverage could never count it.
            report.error(f"{record.id}: venues not in the corpus: {', '.join(sorted(unknown))}")
        if record.is_answerable and not record.venues:
            report.warn(f"{record.id}: answerable question names no venue")


def _check_source_chunks(
    records: Sequence[QuestionRecord], known_chunk_ids: set[str], report: ValidationReport
) -> None:
    if not known_chunk_ids:
        report.warn(f"no {CHUNKS_FILENAME} found; skipped source_chunk_ids check")
        return
    for record in records:
        missing = [c for c in record.source_chunk_ids if c not in known_chunk_ids]
        if missing:
            # chunk_ids move when the chunking config changes, which invalidates the
            # provenance pointers and leaves the questions valid; hence a warning.
            report.warn(
                f"{record.id}: {formatting.plural(len(missing), 'source_chunk_id')} "
                "not in the current corpus "
                f"(re-chunked since generation?)"
            )
        gold_missing = [c for c in record.gold_chunk_ids if c not in known_chunk_ids]
        if gold_missing:
            report.error(
                f"{record.id}: {formatting.plural(len(gold_missing), 'gold_chunk_id')} "
                "not in the current corpus; "
                f"these labels point at nothing in it"
            )


def check_venue_coverage(
    records: Sequence[QuestionRecord], chunk_venues: dict[str, str], report: ValidationReport
) -> None:
    """A labeled answerable comparison must account for every venue it names.

    `venue_coverage` says, per venue, `covered` / `absent_in_docs` /
    `missing_from_pool`. `covered` must be backed by a gold chunk from that
    venue, and a venue that has a gold chunk must be `covered`: the field is a
    claim about the gold set, so the two cannot disagree. Without it a one-sided
    gold set is indistinguishable from a deliberately one-sided answer.
    """
    for record in records:
        question = as_question(record)
        if not (question.needs_venue_coverage and question.is_judged):
            continue
        gold_venues = {chunk_venues[c] for c in record.gold_chunk_ids if c in chunk_venues}
        coverage = record.venue_coverage
        missing = [v for v in record.venues if v not in coverage]
        if missing:
            report.error(f"{record.id}: venue_coverage missing for {', '.join(missing)}")
        extra = sorted(set(coverage) - set(record.venues))
        if extra:
            report.error(
                f"{record.id}: venue_coverage names venues the question does not: "
                f"{', '.join(extra)}"
            )
        for venue, value in sorted(coverage.items()):
            if value == "covered" and venue not in gold_venues:
                report.error(
                    f"{record.id}: venue_coverage says {venue} is covered, but no gold chunk "
                    f"is from {venue}"
                )
            if value != "covered" and venue in gold_venues:
                report.error(
                    f"{record.id}: venue_coverage says {venue} is {value}, but a gold chunk "
                    f"is from {venue}"
                )


def _check_balance(records: Sequence[QuestionRecord], report: ValidationReport) -> None:
    by_category = Counter[str](r.category for r in records)
    by_qtype = Counter[str](r.qtype for r in records)
    by_venue = Counter(v for r in records for v in r.venues)

    report.stats.total = len(records)
    report.stats.by_category = dict(by_category)
    report.stats.by_qtype = dict(by_qtype)
    report.stats.by_venue = dict(by_venue)
    report.stats.answerable = sum(1 for r in records if r.is_answerable)
    report.stats.unanswerable = sum(1 for r in records if not r.is_answerable)
    report.stats.with_gold = sum(1 for r in records if r.gold_chunk_ids)

    for category, count in sorted(by_category.items()):
        if count < MIN_QUESTIONS_PER_CATEGORY:
            report.warn(
                f"category {category!r} has only {formatting.plural(count, 'question')}; "
                "its breakdown row will be noise "
                f"(want >= {formatting.count(MIN_QUESTIONS_PER_CATEGORY)})"
            )
    if by_venue:
        busiest, most = by_venue.most_common(1)[0]
        quietest, least = by_venue.most_common()[-1]
        if most > LOPSIDED_VENUE_RATIO * max(1, least):
            report.warn(
                f"venue coverage is lopsided: {busiest}={formatting.count(most)} "
                f"vs {quietest}={formatting.count(least)}; "
                f"the aggregate will describe {busiest} more than any other venue"
            )


def _check_near_duplicates(
    records: Sequence[QuestionRecord],
    cfg: Config,
    report: ValidationReport,
    encode_documents: DocumentEncoder,
) -> None:
    """Warns of each question pair at or above the configured cosine similarity.

    Two phrasings of the same information need are one question counted twice:
    it doubles that need's weight in every average without adding information.
    """
    if len(records) < PAIR:
        return
    try:
        vectors = encode_documents([r.question for r in records])
    # What `Embedder.model` raises when the model is neither cached nor downloadable, is
    # malformed, or sentence_transformers is missing: the check cannot run, and says so.
    except UserError as exc:
        report.warn(f"near-duplicate check skipped (no encoder available): {exc.with_hint()}")
        return

    similarity = np.asarray(vectors) @ np.asarray(vectors).T
    threshold = cfg.eval.duplicate_threshold
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if similarity[i, j] >= threshold:
                report.warn(
                    f"near-duplicate ({formatting.metric(float(similarity[i, j]))} "
                    f">= {formatting.own_form(threshold, formatting.CONFIG_THRESHOLD_DECIMALS)}): "
                    f"{records[i].id} / {records[j].id}"
                )


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def longest_shared_run(question: str, chunk: str) -> list[str]:
    """Longest run of consecutive words the question shares with the chunk.

    Quoted spans of the question are cut out first, and no run crosses a cut.
    """
    chunk_words = _words(chunk)
    best: list[str] = []
    for segment in _QUOTED_RE.split(question):
        words = _words(segment)
        # Longest common substring over words, one row of the DP table at a time.
        previous = [0] * (len(chunk_words) + 1)
        for i, word in enumerate(words, 1):
            current = [0] * (len(chunk_words) + 1)
            for j, other in enumerate(chunk_words, 1):
                if word == other:
                    current[j] = previous[j - 1] + 1
                    if current[j] > len(best):
                        best = words[i - current[j] : i]
            previous = current
    return best


def check_ngram_leakage(
    records: Sequence[QuestionRecord], chunk_texts: dict[str, str], report: ValidationReport
) -> None:
    """Warns on a question that shares LEAKAGE_NGRAM or more words with a gold chunk."""
    for record in records:
        worst: tuple[list[str], str] = ([], "")
        for chunk_id in record.gold_chunk_ids:
            run = longest_shared_run(record.question, chunk_texts.get(chunk_id, ""))
            if len(run) > len(worst[0]):
                worst = (run, chunk_id)
        run, chunk_id = worst
        if len(run) >= LEAKAGE_NGRAM:
            report.warn(
                f"{record.id}: shares {formatting.plural(len(run), 'consecutive word')} "
                "with gold chunk "
                f'{chunk_id}: "{" ".join(run)}"'
            )


def _check_meta(
    dataset_path: Path,
    records: Sequence[QuestionRecord],
    meta: DatasetMeta,
    cfg: Config,
    report: ValidationReport,
) -> None:
    """The meta against the file, the corpus and the journal it describes."""
    name = meta_path(dataset_path).name
    journal = rounds_path(dataset_path)
    _check_meta_counts(name, records, meta, report)
    _check_meta_snapshot(name, records, meta, cfg, report)
    if journal.exists():
        _check_journal_gold(journal, records, report)
    if meta.hashed:
        _check_meta_hashes(
            name, meta, (("dataset_sha256", dataset_path), ("journal_sha256", journal)), report
        )


def _check_meta_counts(
    name: str, records: Sequence[QuestionRecord], meta: DatasetMeta, report: ValidationReport
) -> None:
    """Each count the meta declares, against the counts `_check_balance` put in `report.stats`."""
    stats = report.stats
    recomputed: dict[str, int | dict[str, int]] = {
        "total": stats.total,
        "by_category": stats.by_category,
        "by_qtype": stats.by_qtype,
        "by_venue": stats.by_venue,
        "answerable": stats.answerable,
        "unanswerable": stats.unanswerable,
        "with_gold": stats.with_gold,
        "gold_chunks": sum(len(r.gold_chunk_ids) for r in records),
    }
    for key, value in recomputed.items():
        declared = getattr(meta.counts, key)
        if declared is not None and declared != value:
            report.error(
                f"{name}: counts.{key} is {_counts_text(declared)}, "
                f"the file gives {_counts_text(value)}"
            )


def _counts_text(value: int | dict[str, int]) -> str:
    """A count, or a breakdown as `comparison=1, fact=2`, as a message shows it."""
    if isinstance(value, int):
        return formatting.count(value)
    return ", ".join(f"{key}={formatting.count(n)}" for key, n in sorted(value.items()))


def _check_meta_snapshot(
    name: str,
    records: Sequence[QuestionRecord],
    meta: DatasetMeta,
    cfg: Config,
    report: ValidationReport,
) -> None:
    """The manifest and index the meta was made against are the current ones.

    A different manifest warns; a different index is an error once the file has
    gold, because chunk ids from two indexes are not comparable.
    """
    snapshot = meta.corpus_snapshot
    if cfg.paths.manifest.exists():
        current = manifest_fingerprint(cfg.paths.manifest).manifest_sha256
        if snapshot.manifest_sha256 != current:
            report.warn(
                f"{name}: made against manifest "
                f"{formatting.short_hash(snapshot.manifest_sha256)}; "
                f"the current {cfg.paths.manifest} has sha256 {current}"
            )
    built = built_index_version(cfg)
    recorded = snapshot.index_version
    if recorded and built and recorded != built:
        issue = report.error if any(r.gold_chunk_ids for r in records) else report.warn
        issue(f"{name}: labeled against index {recorded}, the built index is {built}")


def _check_journal_gold(
    journal_path: Path, records: Sequence[QuestionRecord], report: ValidationReport
) -> None:
    """Every question's gold equals what its journal entries resolve to.

    That is round 1, unless an adjudication or a relabeling after round 2 changed it
    (`QuestionJournal.resolved_relevant`). A journal that cannot be read, or is not a
    journal, is reported as an error, as a damaged meta file is.
    """
    name = journal_path.name
    try:
        journal = read_journal(journal_path)
    except UserError as exc:
        report.error(str(exc))
        return
    for record in records:
        entries = journal.get(record.id)
        if entries.first is None:
            if record.gold_chunk_ids:
                report.error(f"{record.id}: gold chunks but no round-1 entry in {name}")
        elif entries.resolved_relevant() != list(record.gold_chunk_ids):
            report.error(f"{record.id}: gold_chunk_ids differ from what {name} resolves to")


def _check_meta_hashes(
    name: str,
    meta: DatasetMeta,
    hashes: Sequence[tuple[str, Path]],
    report: ValidationReport,
) -> None:
    """Each file the meta hashes has that hash, and a hashed version records every hash."""
    for field_name, file in hashes:
        want = getattr(meta, field_name)
        if want is None and file.exists():
            report.error(f"{name}: no {field_name}; a hashed version records it")
        elif want is not None and not file.exists():
            report.error(f"{name}: {file.name} is missing, and {field_name} records its sha256")
        elif want is not None and file.is_dir():
            report.error(f"{name}: {file.name} is a directory, and {field_name} hashes a file")
        elif want is not None and file_sha256(file) != want:
            report.error(f"{name}: {file.name} does not match {field_name}")


def validate_dataset(
    path: str | Path,
    cfg: Config,
    *,
    check_near_duplicates: bool,
    encode_documents: DocumentEncoder | None = None,
) -> ValidationReport:
    """The validation report of one dataset file against the current corpus and config.

    `encode_documents` encodes the questions for the near-duplicate check; the
    configured `Embedder` unless given. A dataset file or metadata file that cannot be
    read is an error in the report.
    """
    report = ValidationReport()

    try:
        records = load_records(path)
    except UserError as exc:
        report.error(exc.with_hint())
        return report
    if not records:
        report.error(f"{path} contains no questions")
        return report

    _check_records(records, cfg, report)
    if check_near_duplicates:
        encode = encode_documents or Embedder(cfg.embedding).encode_documents
        _check_near_duplicates(records, cfg, report, encode)
    _check_meta_file(Path(path), records, cfg, report)
    return report


def _check_records(
    records: Sequence[QuestionRecord], cfg: Config, report: ValidationReport
) -> None:
    """The per-question checks: ids, notes, duplicates, venues, chunks, balance, leakage."""
    manifest_rows = read_manifest(cfg.paths.manifest)
    corpus_venues = {row.venue for row in manifest_rows}
    if not corpus_venues:
        report.warn(f"{cfg.paths.manifest} is empty; skipped venue check")

    # Without a corpus (CI ingests none), the checks against it are skipped with a warning.
    corpus_built = chunks_path(cfg).exists()
    chunk_venues = load_chunk_venues(cfg) if corpus_built else {}
    known_chunk_ids = set(chunk_venues)

    _check_ids(records, report)
    _check_note_cross_references(records, report)
    _check_exact_duplicates(records, report)
    if corpus_venues:
        _check_venues(records, corpus_venues, report)
    _check_source_chunks(records, known_chunk_ids, report)
    if chunk_venues:
        check_venue_coverage(records, chunk_venues, report)
    _check_balance(records, report)
    if corpus_built and any(r.gold_chunk_ids for r in records):
        check_ngram_leakage(records, read_chunk_field(chunks_path(cfg), "text"), report)


def _check_meta_file(
    dataset_path: Path, records: Sequence[QuestionRecord], cfg: Config, report: ValidationReport
) -> None:
    """The metadata file beside a dataset: parsed, then checked by `_check_meta`."""
    meta_file = meta_path(dataset_path)
    if not meta_file.exists():
        if DATASETS_DIR.name in dataset_path.parts:
            # Only a versioned snapshot needs a metadata file. The working question
            # file is not one: it is written from a selection and its provenance is
            # the candidate pool plus selection.jsonl.
            report.warn(f"no metadata file beside the dataset (expected {meta_file.name})")
        return
    try:
        meta = read_json_model(meta_file, DatasetMeta, hint=META_HINT)
    except UserError as exc:
        report.error(exc.with_hint())
    else:
        _check_meta(dataset_path, records, meta, cfg, report)

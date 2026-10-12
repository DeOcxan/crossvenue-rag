"""Tests for `crossvenue_rag.dataset.validate`."""

from __future__ import annotations

import json
from pathlib import Path

from crossvenue_rag.corpus.manifest import ManifestRow, write_manifest
from crossvenue_rag.dataset.records import QuestionRecord
from crossvenue_rag.dataset.validate import (
    ValidationReport,
    check_venue_coverage,
    validate_dataset,
)
from tests.dataset.factories import record, save_records


def _corpus(cfg, chunk_ids: list[str]) -> None:
    write_manifest(
        cfg.paths.manifest,
        [
            ManifestRow(
                venue=venue,
                venue_type="cex",
                source_url="https://example.invalid/a.md",
                local_path=f"{venue}/a.md",
                fetched_at="2026-01-01T00:00:00Z",
                commit_sha="abc",
                sha256="dead",
            )
            for venue in ("bybit", "binance")
        ],
    )
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    with (cfg.paths.processed / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk_id in chunk_ids:
            fh.write(json.dumps({"chunk_id": chunk_id}) + "\n")


VENUES = {"c1": "binance", "c2": "bybit", "c3": "binance"}


def coverage_record(**kw) -> QuestionRecord:
    base = {
        "id": "cv-0001",
        "question": "how do binance and bybit differ here",
        "tier": "silver",
        "venues": ["binance", "bybit"],
        "qtype": "comparison",
        "category": "comparison",
        "difficulty": "easy",
        "is_answerable": True,
        "expected_behavior": "answer",
        "annotated_at": "2026-09-24T21:00:00Z",
    }
    return QuestionRecord(**{**base, **kw})


def errors(*records: QuestionRecord) -> list[str]:
    report = ValidationReport()
    check_venue_coverage(records, VENUES, report)
    return report.errors


def test_validation_catches_what_silently_corrupts_an_eval(cfg, tmp_path: Path):
    _corpus(cfg, ["c1", "c2"])
    path = tmp_path / "v0.1-candidates.jsonl"
    save_records(
        path,
        [
            record(id="q1", gold_chunk_ids=["c1"]),
            record(id="q1", question="A different question about rate limits entirely?"),
            record(id="q3", venues=["kraken"]),
            record(id="q4", gold_chunk_ids=["gone"]),
            record(id="q5", source_chunk_ids=["gone"]),
        ],
    )
    report = validate_dataset(path, cfg, check_near_duplicates=False)

    assert not report.ok
    joined = " | ".join(report.errors)
    assert "duplicate id 'q1'" in joined
    assert "not in the corpus: kraken" in joined
    assert "gold_chunk_id not in the current corpus" in joined
    # A stale *source* pointer is provenance drift, so it is a warning.
    assert any("source_chunk_id" in w for w in report.warnings)
    assert report.stats.total == 5


def test_notes_may_not_point_at_a_question_that_is_not_there(cfg, tmp_path: Path):
    """Annotators follow these pointers while judging. A dangling one points
    nowhere; a pointer kept through a renumbering silently describes a different
    question. Both happened in v0.1-candidates and neither was caught."""
    _corpus(cfg, ["c1"])
    path = tmp_path / "v0.1-candidates.jsonl"
    save_records(
        path,
        [
            record(id="cv-0001", notes="The answerable sibling is cv-0002."),
            record(id="cv-0002", question="Does Bybit cap recvWindow at the same value?"),
            record(
                id="cv-0003",
                question="What is the BingX recvWindow ceiling in milliseconds?",
                notes="Contrast with cv-0999, which was dropped at selection.",
            ),
            record(
                id="cv-0004",
                question="How does Binance report the used request weight?",
                notes="See cv-0004 for the inverse case.",
            ),
            # The case that slipped through: the target exists, but the set was
            # renumbered and the pointer then names a question of an unrelated venue.
            record(
                id="cv-0005",
                question="What rounds Hyperliquid order prices, tick size or decimals?",
                venues=["hyperliquid"],
                notes="Note the answerable sibling cv-0002 asks the same need.",
            ),
        ],
    )
    report = validate_dataset(path, cfg, check_near_duplicates=False)

    assert not report.ok
    assert any("cv-0003: notes reference cv-0999" in e for e in report.errors)
    # A live pointer is fine; a self-reference is merely useless.
    assert not any("cv-0001" in e for e in report.errors)
    assert any("cv-0004: notes reference the question itself" in w for w in report.warnings)
    assert any(
        w.startswith("cv-0005") and "shares no venue with it; check that the pointer" in w
        for w in report.warnings
    )


def test_validation_passes_a_clean_dataset(cfg, tmp_path: Path):
    _corpus(cfg, ["c1", "c2"])
    path = tmp_path / "v0.1-candidates.jsonl"
    save_records(
        path,
        [
            record(id=f"q{i}", question=f"Question number {i} about Bybit order placement?")
            for i in range(6)
        ],
    )
    report = validate_dataset(path, cfg, check_near_duplicates=False)
    assert report.ok, report.errors
    assert report.stats.by_category == {"fact": 6}


def test_validator_accepts_consistent_coverage():
    assert not errors(
        coverage_record(
            gold_chunk_ids=["c1"],
            venue_coverage={"binance": "covered", "bybit": "missing_from_pool"},
        )
    )


def test_validator_requires_coverage_for_every_venue():
    [message] = errors(
        coverage_record(gold_chunk_ids=["c1"], venue_coverage={"binance": "covered"})
    )
    assert "venue_coverage missing for bybit" in message
    assert errors(coverage_record(gold_chunk_ids=["c1", "c2"]))  # none at all


def test_validator_rejects_covered_without_a_gold_chunk_and_the_reverse():
    [message] = errors(
        coverage_record(
            gold_chunk_ids=["c1"], venue_coverage={"binance": "covered", "bybit": "covered"}
        )
    )
    assert "says bybit is covered, but no gold chunk is from bybit" in message
    [message] = errors(
        coverage_record(
            gold_chunk_ids=["c1", "c2"],
            venue_coverage={"binance": "covered", "bybit": "absent_in_docs"},
        )
    )
    assert "says bybit is absent_in_docs, but a gold chunk is from bybit" in message


def test_validator_skips_unlabelled_and_unanswerable_comparisons():
    assert not errors(coverage_record(annotated_at=""))
    assert not errors(
        coverage_record(
            category="unanswerable",
            is_answerable=False,
            expected_behavior="refuse",
        )
    )


def test_ngram_leakage_flags_a_question_that_restates_its_gold_chunk():
    from crossvenue_rag.dataset.validate import (
        LEAKAGE_NGRAM,
        ValidationReport,
        check_ngram_leakage,
    )

    chunk = "The request weight of this endpoint depends on the number of symbols requested."
    leaky = record(
        id="q1",
        question="does the request weight of this endpoint depend on it",
        gold_chunk_ids=["c1"],
    )
    clean = record(id="q2", question="how heavy is a ticker call on binance", gold_chunk_ids=["c1"])
    report = ValidationReport()
    check_ngram_leakage([leaky, clean], {"c1": chunk}, report)
    assert len(report.warnings) == 1
    assert "q1" in report.warnings[0] and str(LEAKAGE_NGRAM) in report.warnings[0]


def test_ngram_leakage_exempts_quoted_error_strings():
    from crossvenue_rag.dataset.validate import longest_shared_run

    chunk = "Timestamp for this request is outside of the recvWindow."
    question = 'why do i get "Timestamp for this request is outside of the recvWindow" here'
    assert longest_shared_run(question, chunk) == []


def _meta(path: Path, version: str, **patch) -> None:
    import hashlib

    meta = {
        "dataset_version": version,
        "created_at": "2026-01-01T00:00:00Z",
        "corpus_snapshot": {
            "manifest_path": "data/manifest.csv",
            "manifest_sha256": "0" * 64,
            "documents": 2,
            "recorded_at": "2026-01-01T00:00:00Z",
        },
        "sources": [],
        "generator": {},
        "counts": {"total": 2, "by_category": {"fact": 2}},
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **patch,
    }
    meta_path = path.parent / (path.name.removesuffix(".jsonl") + ".meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def test_the_meta_is_checked_against_the_file_its_journal_and_its_hash(cfg, tmp_path: Path):
    _corpus(cfg, ["c1", "c2"])
    path = tmp_path / "v9-gold.jsonl"
    save_records(
        path,
        [
            record(
                id="q1", question="Question one about Bybit order placement?", gold_chunk_ids=["c2"]
            ),
            record(id="q2", question="Question two about Bybit order placement?"),
        ],
    )
    journal = tmp_path / "annotations" / "v9-gold.rounds.json"
    journal.parent.mkdir()
    journal.write_text(json.dumps({"q1": {"1": {"pool": ["c2"], "relevant": ["c2"]}}}))
    import hashlib

    journal_sha = hashlib.sha256(journal.read_bytes()).hexdigest()
    _meta(path, "v9-gold", journal_sha256=journal_sha)
    assert validate_dataset(path, cfg, check_near_duplicates=False).ok

    _meta(path, "v9-gold", journal_sha256=journal_sha, counts={"total": 3})
    assert "counts.total is 3, the file gives 2" in " ".join(
        validate_dataset(path, cfg, check_near_duplicates=False).errors
    )

    _meta(path, "v9-gold", journal_sha256=journal_sha, counts={"by_category": {"fact": 1}})
    assert "counts.by_category is fact=1, the file gives fact=2" in " ".join(
        validate_dataset(path, cfg, check_near_duplicates=False).errors
    )

    journal.write_text(json.dumps({"q1": {"1": {"pool": ["c2"], "relevant": []}}}))
    errors = " ".join(validate_dataset(path, cfg, check_near_duplicates=False).errors)
    assert "q1: gold_chunk_ids differ from what v9-gold.rounds.json resolves to" in errors
    assert "v9-gold.rounds.json does not match journal_sha256" in errors

    _meta(path, "v9-gold", dataset_sha256=None, journal_sha256=journal_sha)
    assert "no dataset_sha256; a hashed version records it" in " ".join(
        validate_dataset(path, cfg, check_near_duplicates=False).errors
    )

    # The earlier format records no hash, so neither file is hashed.
    _meta(path, "v9-gold", dataset_sha256=None)
    assert "sha256" not in " ".join(validate_dataset(path, cfg, check_near_duplicates=False).errors)

    _meta(path, "v9-gold", journal_sha256=journal_sha)
    journal.unlink()
    errors = " ".join(validate_dataset(path, cfg, check_near_duplicates=False).errors)
    assert "v9-gold.rounds.json is missing, and journal_sha256 records its sha256" in errors


def test_a_damaged_or_misshapen_journal_is_a_validation_error_not_a_crash(cfg, tmp_path: Path):
    import pytest

    _corpus(cfg, ["c1", "c2"])
    path = tmp_path / "v9-gold.jsonl"
    save_records(
        path,
        [
            record(
                id="q1", question="Question one about Bybit order placement?", gold_chunk_ids=["c2"]
            ),
            record(id="q2", question="Question two about Bybit order placement?"),
        ],
    )
    _meta(path, "v9-gold")
    journal = tmp_path / "annotations" / "v9-gold.rounds.json"
    journal.parent.mkdir()
    cases = [
        (b"{not json", "v9-gold.rounds.json: Expecting property name"),
        (b"\xff", "v9-gold.rounds.json: 'utf-8' codec can't decode"),
        (b"[1, 2]", "v9-gold.rounds.json: holds a JSON list where an object of rounds"),
        (b'{"q1": [1]}', "v9-gold.rounds.json: q1: Input should be a valid dictionary"),
        (b'{"q1": {"1": 7}}', "v9-gold.rounds.json: q1.rounds.1: Input should be a valid dict"),
        (b'{"q1": {"1": {"relevant": 7}}}', "v9-gold.rounds.json: q1.rounds.1.relevant: Input"),
    ]
    for content, message in cases:
        journal.write_bytes(content)
        report = validate_dataset(path, cfg, check_near_duplicates=False)
        if not any(message in e for e in report.errors):
            pytest.fail(f"{content!r}: no error containing {message!r} in {report.errors}")
    journal.unlink()
    journal.mkdir()
    _meta(path, "v9-gold", journal_sha256="0" * 64)
    errors = " ".join(validate_dataset(path, cfg, check_near_duplicates=False).errors)
    assert "v9-gold.rounds.json is a directory, and journal_sha256 hashes a file" in errors


def test_the_near_duplicate_check_is_skipped_with_a_warning_without_an_encoder(cfg, tmp_path: Path):
    from crossvenue_rag.errors import UserError

    def unavailable(texts):
        raise UserError("cannot load embedding.model_name m: not in the cache")

    _corpus(cfg, ["c1", "c2"])
    path = tmp_path / "v0.1-candidates.jsonl"
    save_records(
        path,
        [
            record(id=f"q{i}", question=f"Question number {i} about Bybit order placement?")
            for i in range(6)
        ],
    )
    report = validate_dataset(path, cfg, check_near_duplicates=True, encode_documents=unavailable)
    assert report.ok, report.errors
    assert any("near-duplicate check skipped" in w for w in report.warnings)


def test_a_dataset_that_cannot_be_read_is_an_error_with_the_loader_hint(cfg, tmp_path: Path):
    report = validate_dataset(tmp_path / "missing.jsonl", cfg, check_near_duplicates=False)
    assert report.errors == [f"{tmp_path / 'missing.jsonl'} not found (check the path of the file)"]


def test_a_meta_field_the_schema_rejects_is_named_without_the_input_value(cfg, tmp_path: Path):
    _corpus(cfg, ["c1", "c2"])
    path = tmp_path / "datasets" / "v0.1-candidates.jsonl"
    save_records(path, [record(id="q1")])
    meta = path.with_suffix(".meta.json")
    meta.write_text(json.dumps({"dataset_version": 5, "created_at": "t"}), encoding="utf-8")
    errors = validate_dataset(path, cfg, check_near_duplicates=False).errors
    [problem] = [e for e in errors if e.startswith(f"invalid {meta}: ")]
    assert "dataset_version: " in problem
    assert "input_value" not in problem and "https://" not in problem
    assert problem.endswith("(restore the .meta.json the dataset version was published with)")

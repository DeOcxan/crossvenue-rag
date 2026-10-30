"""Tests for `crossvenue_rag.dataset.remap`: a journal or dataset carried through an id map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.dataset.remap import RemapRequest, remap_file, remap_rounds
from crossvenue_rag.errors import UserError


def test_remap_rounds_stamps_the_version_and_keeps_what_was_shown(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    path = tmp_path / "q.rounds.json"
    path.write_text(json.dumps({"q1": {"1": {"pool": ["a", "b", "c"], "relevant": ["a", "b"]}}}))
    remap_rounds(
        path, IdMap({"a": "N", "b": "N", "c": "M"}, old_chunks_sha256="", to_index_version=""), "v2"
    )
    entry = json.loads(path.read_text())["q1"]["1"]
    assert entry["pool"] == ["N", "M"]
    assert entry["relevant"] == ["N"]
    assert entry["index_version"] == "v2"
    assert entry["migrated_from"] == {
        "pool": ["a", "b", "c"],
        "relevant": ["a", "b"],
        "index_version": None,
    }


def test_remap_rounds_refuses_an_unmapped_id_and_leaves_the_sidecar_alone(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    path = tmp_path / "q.rounds.json"
    original = json.dumps({"q1": {"1": {"pool": ["a", "dangling"], "relevant": ["dangling"]}}})
    path.write_text(original)
    with pytest.raises(UserError, match=r"1 id.*not in the map: dangling"):
        remap_rounds(path, IdMap({"a": "N"}, old_chunks_sha256="", to_index_version=""), "v2")
    assert path.read_text() == original


def saved_map(tmp_path: Path, version: str = "v2") -> Path:
    from crossvenue_rag.index.migrate import IdMap

    path = tmp_path / "map.json"
    IdMap({"a": "N", "b": "N"}, to_index_version=version, old_chunks_sha256="").save(path)
    return path


def test_remap_file_writes_a_dataset_to_a_new_path_with_the_default_fields(tmp_path: Path):
    source = tmp_path / "in.jsonl"
    records = [
        {"id": "q1", "source_chunk_ids": ["a"], "gold_chunk_ids": ["b"]},
        {"id": "q2", "source_chunk_ids": ["b"], "gold_chunk_ids": []},
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    target = tmp_path / "out.jsonl"
    report, written = remap_file(
        RemapRequest(saved_map(tmp_path), source, target, only=["q1"], fields=None), "v2"
    )
    assert written == target
    out = [json.loads(line) for line in target.read_text().splitlines()]
    assert out[0] == {"id": "q1", "source_chunk_ids": ["N"], "gold_chunk_ids": ["N"]}
    assert out[1] == records[1]  # not in `only`: copied verbatim
    assert report.records == 2 and report.records_changed == 1


def test_remap_file_rewrites_a_sidecar_in_place(tmp_path: Path):
    sidecar = tmp_path / "q.rounds.json"
    sidecar.write_text(json.dumps({"q1": {"1": {"pool": ["a"], "relevant": []}}}))
    report, written = remap_file(
        RemapRequest(saved_map(tmp_path), sidecar, target=None, fields=None, only=()), "v2"
    )
    assert written == sidecar
    assert json.loads(sidecar.read_text())["q1"]["1"]["index_version"] == "v2"
    assert report.records_changed == 1


def test_remap_file_refuses_a_map_for_another_index(tmp_path: Path):
    request = RemapRequest(
        saved_map(tmp_path), tmp_path / "in.jsonl", tmp_path / "out.jsonl", fields=None, only=()
    )
    with pytest.raises(UserError, match="the map targets index v2, the built index is v3"):
        remap_file(request, "v3")


def test_remap_file_needs_a_new_target_for_a_dataset(tmp_path: Path):
    source = tmp_path / "in.jsonl"
    source.write_text("")
    with pytest.raises(UserError, match="needs a target path") as missing:
        remap_file(
            RemapRequest(saved_map(tmp_path), source, target=None, fields=None, only=()), "v2"
        )
    assert (missing.value.hint or "").startswith("pass the target")
    existing = tmp_path / "out.jsonl"
    existing.write_text("")
    with pytest.raises(UserError, match="exists") as info:
        remap_file(RemapRequest(saved_map(tmp_path), source, existing, fields=None, only=()), "v2")
    assert "immutable" in (info.value.hint or "")


@pytest.mark.parametrize(
    ("target", "fields", "refusal"),
    [
        ("out.jsonl", None, "the target .* would not be written"),
        (None, ["gold_chunk_ids"], "--field"),
    ],
)
def test_remap_file_refuses_a_target_or_fields_for_a_journal(
    tmp_path: Path, target, fields, refusal
):
    sidecar = tmp_path / "q.rounds.json"
    original = json.dumps({"q1": {"1": {"pool": ["a"], "relevant": []}}})
    sidecar.write_text(original)
    request = RemapRequest(
        saved_map(tmp_path), sidecar, None if target is None else tmp_path / target, fields, only=()
    )
    with pytest.raises(UserError, match=refusal):
        remap_file(request, "v2")
    assert sidecar.read_text() == original


def _map(mapping: dict[str, str]):
    from crossvenue_rag.index.migrate import IdMap

    return IdMap(mapping, old_chunks_sha256="", to_index_version="")


def _round1() -> dict:
    """Round 1 with a review correction and an augmentation, on the old index `v1`."""
    return {
        "annotator": "me", "annotated_at": "t0", "index_version": "v1",
        "pool": ["a", "c"], "relevant": ["c"],
        "corrections": [{
            "at": "t1", "annotator": "me", "index_version": "v1",
            "previous_relevant": ["a"], "previous_venue_coverage": {},
            "relevant": ["c"], "venue_coverage": {},
        }],
        "augmentations": [{
            "at": "t2", "annotator": "me", "index_version": "v1", "source": "run",
            "shown": ["b"], "relevant": ["b"], "coverage_changes": [],
        }],
    }  # fmt: skip


def test_remap_rounds_carries_corrections_and_augmentations(tmp_path: Path):
    from crossvenue_rag.dataset.journal import read_journal

    path = tmp_path / "q.rounds.json"
    path.write_text(json.dumps({"q1": {"1": _round1()}}))
    report = remap_rounds(path, _map({"a": "A", "b": "B", "c": "C"}), "v2")
    first = read_journal(path).require_first("q1")
    assert first.judged_ids() == ["A", "C", "B"]
    assert first.first_pass_relevant() == {"A", "B"}
    corrections, augmentations = first.corrections or [], first.augmentations or []
    assert corrections[0].previous_relevant == ["A"] and corrections[0].index_version == "v2"
    assert augmentations[0].shown == ["B"] and augmentations[0].index_version == "v2"
    before = first.migrated_from
    assert before is not None and before.index_version == "v1"
    assert (before.augmentations or [])[0].shown == ["b"]
    assert (before.corrections or [])[0].previous_relevant == ["a"]
    assert not report.unmapped and report.ids_before == report.ids_after == 7


def test_remap_rounds_reports_an_unmapped_augmentation_id(tmp_path: Path):
    path = tmp_path / "q.rounds.json"
    original = json.dumps({"q1": {"1": _round1()}})
    path.write_text(original)
    with pytest.raises(UserError, match=r"1 id.*not in the map: b"):
        remap_rounds(path, _map({"a": "A", "c": "C"}), "v2")
    assert path.read_text() == original


def test_a_second_remap_keeps_the_first_migration(tmp_path: Path):
    from crossvenue_rag.dataset.journal import read_journal

    path = tmp_path / "q.rounds.json"
    path.write_text(json.dumps({"q1": {"1": _round1()}}))
    remap_rounds(path, _map({"a": "A", "b": "B", "c": "C"}), "v2")
    remap_rounds(path, _map({"A": "A2", "B": "B2", "C": "C2"}), "v3")
    first = read_journal(path).require_first("q1")
    assert first.pool == ["A2", "C2"] and first.index_version == "v3"
    second = first.migrated_from
    assert second is not None and second.pool == ["A", "C"] and second.index_version == "v2"
    original = second.migrated_from
    assert original is not None and original.pool == ["a", "c"]
    assert original.index_version == "v1" and original.migrated_from is None


def test_remap_rounds_carries_the_adjudication_and_the_relabel(tmp_path: Path):
    from crossvenue_rag.dataset.journal import read_journal

    path = tmp_path / "q.rounds.json"
    adjudicated = {
        "annotator": "me",
        "criterion": "c",
        "index_version": "v1",
        "decisions": {"a": {"relevant": True, "annotated_at": "t"}},
    }
    relabeled = {
        "annotator": "me", "annotated_at": "t", "criterion": "c", "index_version": "v1",
        "pool": ["a", "b"], "relevant": ["b"],
    }  # fmt: skip
    path.write_text(json.dumps({"q1": {"1": _round1(), "adjudicated": adjudicated},
                                "q2": {"1": _round1(), "relabeled": relabeled}}))  # fmt: skip
    remap_rounds(path, _map({"a": "A", "b": "B", "c": "C"}), "v2")
    journal = read_journal(path)
    adjudication = journal.get("q1").adjudicated
    assert adjudication is not None and list(adjudication.decisions) == ["A"]
    assert adjudication.index_version == "v2"
    assert adjudication.migrated_from is not None
    assert list(adjudication.migrated_from.decisions) == ["a"]
    relabel = journal.get("q2").relabeled
    assert relabel is not None and relabel.pool == ["A", "B"] and relabel.relevant == ["B"]
    assert relabel.migrated_from is not None and relabel.migrated_from.pool == ["a", "b"]
    assert journal.get("q1").resolved_relevant() == ["C", "A"]


def test_remap_rounds_refuses_to_merge_decisions_that_differ(tmp_path: Path):
    path = tmp_path / "q.rounds.json"
    adjudicated = {
        "annotator": "me", "criterion": "c", "index_version": "v1",
        "decisions": {"a": {"relevant": True, "annotated_at": "t"},
                      "b": {"relevant": False, "annotated_at": "t"}},
    }  # fmt: skip
    original = json.dumps({"q1": {"1": _round1(), "adjudicated": adjudicated}})
    path.write_text(original)
    with pytest.raises(UserError, match="decided differently: q1 N"):
        remap_rounds(path, _map({"a": "N", "b": "N", "c": "C"}), "v2")
    assert path.read_text() == original


def test_the_committed_gold_journals_are_written_back_byte_identical():
    from crossvenue_rag.dataset.journal import Journal

    for name in ("v0.2-gold", "v0.2.1-gold", "v0.2.2-gold"):
        text = Path(f"evals/datasets/annotations/{name}.rounds.json").read_text(encoding="utf-8")
        assert Journal.from_json(json.loads(text)).to_text() == text


def test_remap_file_refuses_an_only_id_the_journal_lacks_and_writes_nothing(tmp_path: Path):
    sidecar = tmp_path / "q.rounds.json"
    original = json.dumps({"q1": {"1": {"pool": ["a"], "relevant": []}}})
    sidecar.write_text(original)
    with pytest.raises(UserError, match=r"no record of .* has the id q9"):
        remap_file(
            RemapRequest(saved_map(tmp_path), sidecar, target=None, fields=None, only=["q9"]),
            "v2",
        )
    assert sidecar.read_text() == original


def test_remap_rounds_carries_round_3_with_its_seed_and_flag(tmp_path: Path):
    from crossvenue_rag.dataset.journal import read_journal

    path = tmp_path / "q.rounds.json"
    third = {
        "pool": ["b", "a"], "relevant": [], "index_version": "v1", "shuffle_seed": 7,
        "criterion": "c", "flagged": True,
    }  # fmt: skip
    path.write_text(json.dumps({"q1": {"1": _round1(), "3": third}}))
    remap_rounds(path, _map({"a": "A", "b": "B", "c": "C"}), "v2")
    entry = read_journal(path).get("q1").judgment(3)
    assert entry is not None and entry.pool == ["B", "A"] and entry.index_version == "v2"
    assert entry.shuffle_seed == 7 and entry.criterion == "c" and entry.flagged is True

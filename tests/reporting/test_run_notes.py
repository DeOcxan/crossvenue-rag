"""Tests for `reporting.run_notes`: the notes the registry records beside each committed run."""

from crossvenue_rag.reporting.run_notes import NOTES
from crossvenue_rag.reporting.runs import committed_runs
from tests.conftest import REPO


def test_every_note_belongs_to_a_committed_run_and_is_a_sentence():
    unknown = sorted(set(NOTES) - set(committed_runs(REPO)))
    assert not unknown, f"notes of runs that are not committed: {unknown}"
    unfinished = [(run, n) for run, notes in NOTES.items() for n in notes if not n.endswith(".")]
    assert not unfinished, f"notes that are not whole sentences: {unfinished}"

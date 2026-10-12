"""Tests for `labeling.modes`: the `label` command's option combinations."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.modes import LabelRequest, check_augment, label_plan, review_ids
from tests.conftest import write_journal


def request(**fields) -> LabelRequest:
    return LabelRequest(questions_path="q.jsonl", **fields)


def test_augment_needs_its_candidate_list_and_no_other_mode():
    assert check_augment(request(augment=True, question_file=Path("c.jsonl"))) == Path("c.jsonl")
    with pytest.raises(UserError, match="go together"):
        check_augment(request(augment=True))
    with pytest.raises(UserError, match="go together"):
        check_augment(request(question_file=Path("c.jsonl")))
    for other in ({"ids": ("cv-0001",)}, {"review": True}, {"annotation_round": 2}):
        with pytest.raises(UserError, match="cannot be combined"):
            check_augment(request(augment=True, question_file=Path("c.jsonl"), **other))


def test_review_ids_come_from_arguments_and_the_review_file(tmp_path: Path):
    listing = tmp_path / "review.txt"
    listing.write_text("cv-0002 reason\n", encoding="utf-8")
    assert review_ids(request()) == []
    assert review_ids(request(ids=("cv-0001",), review=True)) == ["cv-0001"]
    assert review_ids(request(ids=("cv-0001",), review_file=listing)) == ["cv-0001", "cv-0002"]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"ids": ("cv-0001",)}, "without --review"),
        ({"review": True}, "needs question ids"),
        ({"sample_file": Path("s.txt")}, "re-annotation subsample"),
        (
            {"sample_file": Path("s.txt"), "annotation_round": 2, "sample": 5},
            "re-annotation subsample",
        ),
        ({"ids": ("cv-0001",), "review": True, "annotation_round": 2}, "corrects round 1"),
    ],
)
def test_mixed_review_options_are_refused(fields: dict, message: str):
    with pytest.raises(UserError, match=message):
        review_ids(request(**fields))


def test_the_label_plan_carries_every_option(tmp_path: Path):
    listing = tmp_path / "sample.txt"
    listing.write_text("cv-0003\n", encoding="utf-8")
    plan = label_plan(
        request(
            revisit=True,
            limit=7,
            annotation_round=2,
            annotator="me",
            seed=3,
            sample_file=listing,
        ),
        "v1",
        30,
    )
    assert (plan.target.questions_path, plan.target.index_version, plan.target.who()) == (
        "q.jsonl",
        "v1",
        "me",
    )
    assert plan.queue.annotation_round == 2 and not plan.queue.only_unlabelled
    assert plan.queue.sample_ids == ["cv-0003"] and plan.queue.seed == 3
    assert plan.per_method == 30 and plan.limit == 7


class VersionedRetriever:
    """Stands in for Retriever: a context manager that has the index version."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.warmed = False

    def __enter__(self) -> VersionedRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def require_current_index(self) -> str:
        return "v9"

    def warmup(self) -> None:
        self.warmed = True


@pytest.fixture
def sessions(monkeypatch) -> dict:
    """The sessions `run_label` dispatched to, each with its arguments, and the
    retrievers it opened (under "opened")."""
    from crossvenue_rag.labeling import modes

    calls: dict = {"opened": []}

    def open_retriever(cfg: Config) -> VersionedRetriever:
        fake = VersionedRetriever(cfg)
        calls["opened"].append(fake)
        return fake

    def recorder(name: str):
        def record(*args) -> None:
            calls[name] = args

        return record

    monkeypatch.setattr(modes, "Retriever", open_retriever)
    monkeypatch.setattr(modes, "augment_session", recorder("augment"))
    monkeypatch.setattr(modes, "review_session", recorder("review"))
    monkeypatch.setattr(modes, "label_session", recorder("label"))
    return calls


def test_augment_runs_the_augment_session_without_warmup(sessions: dict, cfg: Config):
    from crossvenue_rag.labeling.modes import run_label

    run_label(cfg, request(augment=True, question_file=Path("c.jsonl"), annotator="me"))
    [opened] = sessions["opened"]
    retriever, target, candidates = sessions["augment"]
    assert retriever is opened and not opened.warmed
    assert (target.questions_path, target.index_version, target.who()) == ("q.jsonl", "v9", "me")
    assert candidates == Path("c.jsonl")
    assert "review" not in sessions and "label" not in sessions


def test_review_ids_open_the_review_session(sessions: dict, cfg: Config):
    from crossvenue_rag.labeling.modes import run_label

    run_label(cfg, request(ids=("cv-0001", "cv-0002"), review=True))
    [opened] = sessions["opened"]
    got_cfg, retriever, target, ids = sessions["review"]
    assert got_cfg is cfg and retriever is opened and opened.warmed
    assert target.index_version == "v9" and ids == ["cv-0001", "cv-0002"]
    assert "label" not in sessions


def test_no_mode_option_runs_the_label_session_on_the_current_index(sessions: dict, cfg: Config):
    from crossvenue_rag.labeling.modes import run_label

    run_label(cfg, request(annotation_round=2, sample=5, seed=7))
    got_cfg, retriever, plan = sessions["label"]
    assert got_cfg is cfg and retriever is sessions["opened"][0] and retriever.warmed
    assert plan.target.index_version == "v9" and plan.per_method == cfg.eval.pool_depth_per_method
    assert (plan.queue.annotation_round, plan.queue.sample, plan.queue.seed) == (2, 5, 7)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"question_file": Path("c.jsonl"), "review": True}, "go together"),
        (
            {"augment": True, "question_file": Path("c.jsonl"), "annotation_round": 2},
            "cannot be combined",
        ),
        ({"ids": ("cv-0001",)}, "without --review"),
        ({"review": True}, "needs question ids"),
        ({"pool_depth": 1}, "--pool-depth 1: a pool depth of 1 does not exceed"),
    ],
)
def test_a_mixed_request_is_refused_before_the_index_is_opened(
    sessions: dict, fields: dict, message: str, cfg: Config
):
    from crossvenue_rag.labeling.modes import run_label

    with pytest.raises(UserError, match=message):
        run_label(cfg, request(**fields))
    assert sessions["opened"] == []


def test_adjudicate_takes_no_ids_and_no_other_mode():
    from crossvenue_rag.labeling.modes import check_adjudicate

    check_adjudicate(request(adjudicate=True))
    for other, named in (
        ({"ids": ("cv-0001",)}, "cv-0001"),
        ({"review": True}, "--review"),
        ({"annotation_round": 2}, "--round"),
        ({"relabel": True}, "--relabel"),
        ({"add_chunks": Path("a.jsonl")}, "--add-chunks"),
        ({"augment": True}, "--augment"),
    ):
        with pytest.raises(UserError, match=named):
            check_adjudicate(request(adjudicate=True, **other))


def test_relabel_needs_ids_and_add_chunks_needs_relabel():
    from crossvenue_rag.labeling.modes import check_relabel

    assert check_relabel(request(relabel=True, ids=("cv-0106",))) == ["cv-0106"]
    with pytest.raises(UserError, match="needs question ids"):
        check_relabel(request(relabel=True))
    with pytest.raises(UserError, match="goes with --relabel"):
        check_relabel(request(add_chunks=Path("a.jsonl"), ids=("cv-0106",)))
    with pytest.raises(UserError, match="cannot be combined with --sample-file"):
        check_relabel(request(relabel=True, ids=("cv-0106",), sample_file=Path("s.txt")))


def test_the_offline_modes_run_before_any_retriever_is_opened(monkeypatch):
    from crossvenue_rag.labeling import modes

    ran: list = []
    monkeypatch.setattr(modes, "run_offline", lambda cfg, req: ran.append(req))
    for fields in ({"adjudicate": True}, {"relabel": True, "ids": ("q",)}):
        modes.run_label(Config(), request(**fields))
    assert [r.adjudicate for r in ran] == [True, False]


def test_run_offline_checks_the_index_and_hands_the_chunks_to_the_session(tmp_path, monkeypatch):
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.labeling import modes
    from crossvenue_rag.models import Question
    from tests.labeling.conftest import VERSION, chunk

    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact")])
    write_journal(path, {"q1": {"1": {"pool": ["c1"], "relevant": [], "index_version": VERSION}}})
    monkeypatch.setattr(modes, "require_current_index", lambda cfg: VERSION)
    monkeypatch.setattr(modes, "load_chunks", lambda cfg: [chunk("c1", "bybit")])
    seen: list = []
    monkeypatch.setattr(modes, "adjudicate_session", lambda target: seen.append(target))
    monkeypatch.setattr(modes, "relabel_session", lambda target: seen.append(target))
    added = tmp_path / "added.jsonl"
    added.write_text('{"id": "q1", "chunk_ids": ["c1"]}\n', encoding="utf-8")
    adjudicating = LabelRequest(str(path), annotator="me", adjudicate=True)
    modes.run_offline(Config(), adjudicating)
    modes.run_offline(
        Config(),
        LabelRequest(str(path), annotator="me", relabel=True, ids=("q1",), add_chunks=added),
    )
    adjudication, relabeling = seen
    assert adjudication.index_version == VERSION and set(adjudication.chunks) == {"c1"}
    assert relabeling.ids == ["q1"] and relabeling.added == {"q1": ["c1"]}
    write_journal(path, {"q1": {"1": {"pool": ["c1"], "relevant": [], "index_version": "old"}}})
    with pytest.raises(UserError, match="not made against the current index"):
        modes.run_offline(Config(), adjudicating)

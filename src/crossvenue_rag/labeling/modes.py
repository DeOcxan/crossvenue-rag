"""The `label` command's modes, chosen from its options.

`crossvenue label` labels a round (`label_session`), or runs one of the modes `--review`,
`--augment`, `--adjudicate`, `--relabel` and `--round 3`, each with its own session. `run_label`
refuses option combinations that mix modes before any index is opened, then runs the one mode
the options select.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import formatting
from ..config import Config
from ..corpus.artifacts import load_chunks
from ..dataset.added_chunks import read_added_chunks
from ..dataset.journal import FIRST_ROUND, FULL_RELABEL_ROUND
from ..dataset.questions import METADATA_FROM_HINT, load_questions, require_label_versions
from ..dataset.sampling import read_sample_file
from ..errors import UserError, require_file
from ..index_version import require_current_index
from ..retrieval.retriever import Retriever
from .adjudicate import AdjudicationTarget, adjudicate_session
from .augment import augment_session
from .pool import pool_depth_of
from .relabel import RelabelTarget, relabel_session
from .review import review_session
from .round3 import Round3Target, round3_session
from .session import (
    LabelPlan,
    LabelTarget,
    QueueSpec,
    default_annotator,
    label_session,
    select_queue,
)


@dataclass(frozen=True)
class LabelRequest:
    """The `label` command's options, one field per option."""

    questions_path: str
    ids: Sequence[str]
    review: bool
    review_file: Path | None
    augment: bool
    question_file: Path | None
    revisit: bool
    limit: int | None
    pool_depth: int | None
    annotation_round: int
    annotator: str
    sample: int | None
    seed: int
    sample_file: Path | None
    adjudicate: bool
    relabel: bool
    add_chunks: Path | None
    metadata_from: Path | None

    @property
    def augments(self) -> bool:
        """Whether any `--augment` option was given."""
        return self.augment or self.question_file is not None

    @property
    def other_modes(self) -> list[str]:
        """The options of the round, review and augment modes this request sets."""
        given = {
            "--review": self.review or self.review_file is not None,
            "--augment": self.augments,
            "--round": self.annotation_round != FIRST_ROUND,
            "--sample": self.sample is not None,
            "--sample-file": self.sample_file is not None,
            "--all": self.revisit,
            "--seed": self.seed != 0,
            "--limit": self.limit is not None,
            "--pool-depth": self.pool_depth is not None,
        }
        return [option for option, on in given.items() if on]


def check_adjudicate(request: LabelRequest) -> None:
    """Refuses `--adjudicate` combined with question ids or another mode's options.

    Raises:
        UserError: The request names question ids or another mode's options.
    """
    others = request.other_modes + (["--relabel"] if request.relabel else [])
    others += ["--add-chunks"] if request.add_chunks is not None else []
    if request.ids or others:
        raise UserError(
            "--adjudicate walks every disagreeing pair and takes no ids or other mode: "
            + " ".join(list(request.ids) + others),
            hint="crossvenue label --adjudicate",
        )


def check_relabel(request: LabelRequest) -> list[str]:
    """The ids `--relabel` judges; refuses it without ids or with another mode's options.

    Raises:
        UserError: `--add-chunks` comes with neither `--relabel` nor `--round 3`, no id is
            given, or another mode's options are set.
    """
    if request.add_chunks is not None and not request.relabel:
        raise UserError(
            f"--add-chunks goes with --relabel or --round {formatting.own_form(FULL_RELABEL_ROUND)}"
        )
    if not request.ids:
        raise UserError(
            "--relabel needs question ids", hint="crossvenue label --relabel ID [ID ...]"
        )
    if request.other_modes:
        raise UserError(f"--relabel cannot be combined with {' '.join(request.other_modes)}")
    return list(request.ids)


def check_round3(request: LabelRequest) -> tuple[Path, Path]:
    """The added-chunks file and metadata file of `--round 3`; refuses anything else.

    Raises:
        UserError: `--metadata-from` comes without `--round 3`, a required option is
            missing, or another mode's option is set.
    """
    own = formatting.own_form(FULL_RELABEL_ROUND)
    if request.annotation_round != FULL_RELABEL_ROUND:
        raise UserError(f"--metadata-from goes with --round {own}")
    if request.add_chunks is None or request.metadata_from is None:
        raise UserError(
            f"--round {own} needs --add-chunks and --metadata-from",
            hint="make round3",
        )
    given = {"--adjudicate": request.adjudicate, "--relabel": request.relabel}
    others = [option for option in request.other_modes if option != "--round"]
    others += [option for option, on in given.items() if on]
    if request.ids or others:
        raise UserError(
            f"--round {own} walks every question and takes no ids or other mode: "
            + " ".join(list(request.ids) + others),
            hint="make round3",
        )
    return request.add_chunks, request.metadata_from


def check_augment(request: LabelRequest) -> Path:
    """The candidate list `--augment` judges; refuses it combined with another mode.

    Raises:
        UserError: `--augment` and `--question-file` do not come together, or another
            mode's options are set.
    """
    if not request.augment or request.question_file is None:
        raise UserError(
            "--augment and --question-file go together",
            hint=(
                "crossvenue label --augment --question-file evals/selection/augment_round1.jsonl"
            ),
        )
    others = [option for option in request.other_modes if option != "--augment"]
    if request.ids or others:
        raise UserError(
            "--augment extends round 1 and takes no ids or other mode: "
            + " ".join(list(request.ids) + others)
        )
    return request.question_file


def review_ids(request: LabelRequest) -> list[str]:
    """The ids `--review` re-opens, empty when labeling; refuses a mixed request.

    Raises:
        UserError: Ids come without `--review`, `--review` comes without ids, `--sample-file`
            comes outside a later round or with `--sample`, ids come with another mode's
            options, or a later round comes with `--limit` or `--pool-depth`. Nothing is at
            `request.review_file`, or a directory is.
    """
    ids = list(request.ids)
    if request.review_file is not None:
        ids += read_sample_file(request.review_file)
    reviewing = request.review or request.review_file is not None
    if ids and not reviewing:
        raise UserError(
            f"question ids given without --review: {' '.join(ids)}",
            hint="use `crossvenue label --review ID [ID ...]`",
        )
    if reviewing and not ids:
        named = "" if request.review_file is None else f"; {request.review_file} names none"
        raise UserError(
            f"--review needs question ids{named}",
            hint="pass ids after --review, or one per line in --review-file",
        )
    _check_sample_file(request)
    others = [option for option in request.other_modes if option != "--review"]
    if ids and others:
        raise UserError(f"--review corrects round 1; it cannot be combined with {' '.join(others)}")
    _check_later_round(request)
    return ids


def _check_sample_file(request: LabelRequest) -> None:
    """Refuses `--sample-file` outside a later round or together with `--sample`."""
    if request.sample_file is not None and (
        request.annotation_round == FIRST_ROUND or request.sample is not None
    ):
        raise UserError(
            "--sample-file selects a re-annotation subsample",
            hint="use it with --round 2 and without --sample",
        )


def _check_later_round(request: LabelRequest) -> None:
    """Refuses `--limit` and `--pool-depth` in a later round, whose candidates are fixed."""
    given = {"--limit": request.limit, "--pool-depth": request.pool_depth}
    others = [option for option, value in given.items() if value is not None]
    if request.annotation_round != FIRST_ROUND and others:
        raise UserError(
            f"--round {formatting.own_form(request.annotation_round)} shows the candidates round 1 "
            f"judged and cannot be combined with {' '.join(others)}"
        )


def _queue_spec(request: LabelRequest, sample_ids: Sequence[str] | None) -> QueueSpec:
    return QueueSpec(
        annotation_round=request.annotation_round,
        only_unjudged=not request.revisit,
        sample=request.sample,
        seed=request.seed,
        sample_ids=sample_ids,
    )


def check_sample(request: LabelRequest) -> None:
    """Refuses a `--sample` from which the round's queue cannot be drawn, naming the option.

    A later round stratifies `--sample` by the judged questions' categories, so it needs one
    question per category at least.

    Raises:
        UserError: The queue of `request.sample` questions cannot be drawn. Nothing is at
            `request.questions_path`, or a directory is.
    """
    if request.sample is None:
        return
    questions = load_questions(request.questions_path)
    try:
        select_queue(questions, _queue_spec(request, None))
    except UserError as refused:
        raise UserError(
            f"--sample {formatting.own_form(request.sample)}: {refused}", hint=refused.hint
        ) from refused


def label_plan(request: LabelRequest, index_version: str, per_method: int) -> LabelPlan:
    """The labeling session the options describe, against `index_version`.

    `per_method` is the pool depth `pool.pool_depth_of` returned for `--pool-depth`.

    Raises:
        UserError: Nothing is at `request.sample_file`, or a directory is.
    """
    sample_ids = read_sample_file(request.sample_file) if request.sample_file else None
    return LabelPlan(
        target=LabelTarget(request.questions_path, index_version, request.annotator),
        queue=_queue_spec(request, sample_ids),
        per_method=per_method,
        limit=request.limit,
    )


def run_offline(cfg: Config, request: LabelRequest) -> None:
    """Runs `--adjudicate` or `--relabel` on the chunks file, checked against the built index.

    Raises:
        UserError: `check_adjudicate` or `check_relabel` refuses the options of `request`, or
            nothing is at `request.questions_path` or `request.add_chunks`, or a directory is.
    """
    if request.adjudicate:
        check_adjudicate(request)
        ids: list[str] = []
    else:
        ids = check_relabel(request)
    added = read_added_chunks(request.add_chunks) if request.add_chunks else {}
    version = require_current_index(cfg)
    require_label_versions(request.questions_path, version, all_rounds=True)
    chunks = {c.chunk_id: c for c in load_chunks(cfg)}
    annotator = request.annotator or default_annotator()
    if request.adjudicate:
        adjudicate_session(AdjudicationTarget(request.questions_path, chunks, version, annotator))
        return
    relabel_session(RelabelTarget(request.questions_path, ids, added, chunks, version, annotator))


def run_round3(cfg: Config, request: LabelRequest) -> None:
    """Runs `--round 3` on the chunks file, checked against the built index.

    Raises:
        UserError: `check_round3` refuses the options of `request`, or nothing is at
            `request.questions_path`, `request.add_chunks` or `request.metadata_from`, or a
            directory is.
    """
    added_file, metadata = check_round3(request)
    require_file(metadata, hint=METADATA_FROM_HINT)
    added = read_added_chunks(added_file)
    version = require_current_index(cfg)
    require_label_versions(request.questions_path, version, all_rounds=True)
    require_label_versions(metadata, version)
    target = Round3Target(
        request.questions_path,
        load_questions(metadata),
        added,
        {c.chunk_id: c for c in load_chunks(cfg)},
        version,
        request.annotator or default_annotator(),
    )
    round3_session(target)


def run_label(cfg: Config, request: LabelRequest) -> None:
    """Checks the options, opens the index and runs the mode they select.

    `--adjudicate`, `--relabel` and `--round 3` read chunks from the corpus file and need no
    database.

    Raises:
        UserError: `check_augment`, `review_ids`, `check_sample`, `label_plan`, `check_round3`,
            `check_adjudicate` or `check_relabel` refuses the options of `request`, or
            `request.pool_depth` is not deeper than the deepest rank the metrics score. Or
            nothing is at `request.questions_path`, `request.question_file`,
            `request.add_chunks` or `request.metadata_from`, or a directory is.
    """
    if request.annotation_round == FULL_RELABEL_ROUND or request.metadata_from is not None:
        run_round3(cfg, request)
        return
    if request.adjudicate or request.relabel or request.add_chunks is not None:
        run_offline(cfg, request)
        return
    if request.augments:
        candidates = check_augment(request)
        with Retriever(cfg) as retriever:
            target = LabelTarget(
                request.questions_path, retriever.require_current_index(), request.annotator
            )
            augment_session(retriever, target, candidates)
        return
    ids = review_ids(request)
    check_sample(request)
    per_method = pool_depth_of(cfg, request.pool_depth, "--pool-depth")
    with Retriever(cfg) as retriever:
        version = retriever.require_current_index()
        retriever.warmup()
        if ids:
            target = LabelTarget(request.questions_path, version, request.annotator)
            review_session(cfg, retriever, target, ids)
            return
        label_session(cfg, retriever, label_plan(request, version, per_method))
